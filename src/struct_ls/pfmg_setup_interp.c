/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/* 2: the most explicit implementation, a function for each stencil size */
#define CC0_IMPLEMENTATION 2

/*--------------------------------------------------------------------------
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
   stencil_shape = hypre_CTAlloc(hypre_Index,  stencil_size, HYPRE_MEMORY_HOST);
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
   hypre_StructMatrixSetNumGhost(P, num_ghost);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient == 2 )
   {
      if ( rap_type == 0 )
         /* A has variable diagonal, which will force all P coefficients to be variable */
      {
         hypre_StructMatrixSetConstantCoefficient(P, 0 );
      }
      else
      {
         /* We will force P to be 0.5's everywhere, ignoring A. */
         hypre_StructMatrixSetConstantCoefficient(P, 1);
      }
   }
   else
   {
      /* constant_coefficient = 0 or 1: A is entirely constant or entirely
         variable coefficient */
      hypre_StructMatrixSetConstantCoefficient( P, constant_coefficient );
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
#if CC0_IMPLEMENTATION <= 1
         hypre_PFMGSetupInterpOp_CC0
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
#else
         switch (stencil_size)
         {
            case 5:
               hypre_PFMGSetupInterpOp_CC0_SS5
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 9:
               hypre_PFMGSetupInterpOp_CC0_SS9
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 7:
               hypre_PFMGSetupInterpOp_CC0_SS7
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 15:
               hypre_PFMGSetupInterpOp_CC0_SS15
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 19:
               hypre_PFMGSetupInterpOp_CC0_SS19
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 27:
               hypre_PFMGSetupInterpOp_CC0_SS27
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            default:
               /*
               hypre_PFMGSetupInterpOp_CC0
                  ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                    P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
                */

               hypre_printf("hypre error: unsupported stencil size %d\n", stencil_size);
               hypre_MPI_Abort(hypre_MPI_COMM_WORLD, 1);
         }
#endif
      }
   }

#if 0
   hypre_StructMatrixAssemble(P);
#else
   hypre_StructInterpAssemble(A, P, 0, cdir, findex, stride);
#endif

   return hypre_error_flag;
}

#if CC0_IMPLEMENTATION == 0

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
   hypre_StructStencil *stencil = hypre_StructMatrixStencil(A);
   hypre_Index         *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int            stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int            warning_cnt = 0;
   HYPRE_Int            data_location = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
   HYPRE_Int          **data_indices = hypre_StructMatrixDataIndices(A);
   HYPRE_Complex       *matrixA_data = hypre_StructMatrixData(A);
   HYPRE_Int           *data_indices_boxi_d;
   hypre_Index         *stencil_shape_d;
   HYPRE_MemoryLocation memory_location = hypre_StructMatrixMemoryLocation(A);

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      data_indices_boxi_d = hypre_TAlloc(HYPRE_Int, stencil_size, memory_location);
      stencil_shape_d = hypre_TAlloc(hypre_Index, stencil_size, memory_location);
      hypre_TMemcpy(data_indices_boxi_d, data_indices[i], HYPRE_Int, stencil_size, memory_location,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(stencil_shape_d, stencil_shape, hypre_Index, stencil_size, memory_location,
                    HYPRE_MEMORY_HOST);
   }
   else
   {
      data_indices_boxi_d = data_indices[i];
      stencil_shape_d = stencil_shape;
   }

#define DEVICE_VAR is_device_ptr(Pp0,Pp1,matrixA_data,stencil_shape_d,data_indices_boxi_d)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start,  stride,  Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Int si, mrk0, mrk1, Astenc;
      HYPRE_Real center;
      HYPRE_Real *Ap;

      center  = 0.0;
      Pp0[Pi] = 0.0;
      Pp1[Pi] = 0.0;
      mrk0 = 0;
      mrk1 = 0;

      for (si = 0; si < stencil_size; si++)
      {
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         if (data_location != HYPRE_MEMORY_HOST)
         {
            Ap     = matrixA_data + data_indices_boxi_d[si];
            Astenc = hypre_IndexD(stencil_shape_d[si], cdir);
         }
         else
         {
            Ap     = hypre_StructMatrixBoxData(A, i, si);
            Astenc = hypre_IndexD(stencil_shape[si], cdir);
         }
#else
         Ap     = matrixA_data + data_indices_boxi_d[si];
         Astenc = hypre_IndexD(stencil_shape_d[si], cdir);
#endif

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
         //warning_cnt++;
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
#undef DEVICE_VAR

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

#endif

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

#endif


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
      Ap     = hypre_StructMatrixBoxData(A, i, si);
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
   HYPRE_Int              si;
   HYPRE_Int              Ai;
   HYPRE_Int              Pi;
   HYPRE_Real            *Ap;
   HYPRE_Real             P0, P1;
   HYPRE_Real             center_offd;
   HYPRE_Int              Astenc;
   HYPRE_Int              mrk0_offd, mrk1_offd;
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   hypre_Index            diag_index;
   HYPRE_Int              diag_rank;
   HYPRE_Int              warning_cnt = 0;

   hypre_SetIndex3(diag_index, 0, 0, 0);
   diag_rank = hypre_StructStencilElementRank(stencil, diag_index);

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
      Ai = hypre_CCBoxIndexRank(A_dbox, start );

      center_offd  = 0.0;
      P0 = 0.0;
      P1 = 0.0;
      mrk0_offd = 0;
      mrk1_offd = 0;

      for (si = 0; si < stencil_size; si++)
      {
         if ( si != diag_rank )
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

      si = diag_rank;

      HYPRE_Real *Ap = hypre_StructMatrixBoxData(A, i, si);

#define DEVICE_VAR is_device_ptr(Pp0,Pp1,Ap)
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         HYPRE_Int   mrk0, mrk1;
         HYPRE_Real  center;
         HYPRE_Real  p0val, p1val;

         p0val = P0;
         p1val = P1;
         center = center_offd;
         mrk0 = mrk0_offd;
         mrk1 = mrk1_offd;

         /* RL: Astenc is only needed for assertion, comment out
            Astenc = hypre_IndexD(stencil_shape[si], cdir);
            hypre_assert( Astenc==0 );
         */

         center += Ap[Ai];

         //if (si == si0 && Ap[Ai] == 0.0)
         //   mrk0++;
         //if (si == si1 && Ap[Ai] == 0.0)
         //   mrk1++;

         if (!center)
         {
            //warning_cnt++;
            p0val = 0.0;
            p1val = 0.0;
         }
         else
         {
            p0val /= center;
            p1val /= center;
         }

         /*----------------------------------------------
          * Set interpolation weight to zero, if stencil
          * entry in same direction is zero. Prevents
          * interpolation and operator stencils reaching
          * outside domain.
          *----------------------------------------------*/
         if (mrk0 != 0)
         {
            p0val = 0.0;
         }
         if (mrk1 != 0)
         {
            p1val = 0.0;
         }
         Pp0[Pi] = p0val;
         Pp1[Pi] = p1val;

      }
      hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR
   }

   if (warning_cnt)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return hypre_error_flag;
}

#if CC0_IMPLEMENTATION > 1

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS5
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(rap_type);
   HYPRE_UNUSED_VAR(Pstenc1);

   //hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   //hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   //HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   //HYPRE_Int              warning_cnt= 0;

   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real            *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_cw,a_ce,Pp0,Pp1,p0,p1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] + a_cs[Ai] + a_cn[Ai];
            left   = -a_cw[Ai];
            right  = -a_ce[Ai];
            break;
      case 1: default:
            center = a_cc[Ai] + a_cw[Ai] + a_ce[Ai];
            left   = -a_cs[Ai];
            right  = -a_cn[Ai];
            break;
      }

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               break;
            case  1:
               Pp0[Pi] = right / center;
               break;
         }

         switch (Pstenc1)
         {
            case -1:
               Pp1[Pi] = left / center;
               break;
            case  1:
               Pp1[Pi] = right / center;
               break;
         }
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      /*----------------------------------------------
       * Set interpolation weight to zero, if stencil
       * entry in same direction is zero. Prevents
       * interpolation and operator stencils reaching
       * outside domain.
       *----------------------------------------------*/
      //if (mrk0 != 0)
      //   Pp0[Pi] = 0.0;
      //if (mrk1 != 0)
      //   Pp1[Pi] = 0.0;
   }
   hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS9
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(rap_type);
   HYPRE_UNUSED_VAR(Pstenc1);

   //hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   //hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   //HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   //HYPRE_Int              warning_cnt= 0;

   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real            *a_csw, *a_cse, *a_cne, *a_cnw;
   HYPRE_Real            *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);
   /*-----------------------------------------------------------------
    * Extract pointers for 5-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_cw,a_csw,a_cnw,a_ce,a_cse,a_cne,Pp0,Pp1,p0,p1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] +  a_cn[Ai];
            left   = -a_cw[Ai] - a_csw[Ai] - a_cnw[Ai];
            right  = -a_ce[Ai] - a_cse[Ai] - a_cne[Ai];
            break;
      case 1: default:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai];
            left   = -a_cs[Ai] - a_csw[Ai] - a_cse[Ai];
            right  = -a_cn[Ai] - a_cnw[Ai] - a_cne[Ai];
            break;
      };

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
   }
   hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS7
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(rap_type);
   HYPRE_UNUSED_VAR(Pstenc1);

   //hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   //hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   //HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   //HYPRE_Int              warning_cnt= 0;

   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real            *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_cw,a_ce,Pp0,Pp1,p0,p1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai];
            left   = -a_cw[Ai];
            right  = -a_ce[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] ;
            left   = -a_cs[Ai];
            right  = -a_cn[Ai];
            break;
      case 2: default:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_cs[Ai] + a_cn[Ai] ;
            left   = -a_bc[Ai];
            right  = -a_ac[Ai];
            break;
      };

      if (!center)
      {
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }

      //printf("%d: %d, Pp0[%d] = %e, Pp1 = %e, %e, %e, %e, cc=%e, cw=%e, ce=%e, cs=%e, cn=%e, bc=%e, ac=%e \n",Ai,cdir, Pi,Pp0[Pi],Pp1[Pi],center, left, right,
      //     a_cc[Ai],a_cw[Ai],a_ce[Ai],a_cs[Ai],a_cn[Ai],a_bc[Ai],a_ac[Ai]);
   }
   hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return hypre_error_flag;
}


HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS15
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(rap_type);
   HYPRE_UNUSED_VAR(Pstenc1);

   hypre_Index           index;
   HYPRE_Int             stencil_type15;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 15-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, 0, 1);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 1);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 1);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 1);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, -1);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, -1);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, -1);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, -1);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   if (a_csw)
   {
      if (a_as)
      {
         stencil_type15 = 1;
      }
      else
      {
         stencil_type15 = 0;
      }
   }
   else
   {
      stencil_type15 = 2;
   }

   //printf("loop_size %d %d %d, cdir %d, %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p\n", loop_size[0], loop_size[1], loop_size[2], cdir, a_cc, a_cw, a_ce, a_ac, a_bc, a_cs, a_as, a_bs, a_csw, a_cse, a_cn, a_an, a_bn, a_cnw, a_cne);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_ce,a_ae,a_be,a_cnw,a_cne,a_csw,a_cse,Pp0,Pp1,p0,p1)
   if (stencil_type15 == 0)
   {
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] +  a_bc[Ai];
               left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai];
               right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai];
               break;
            case 1:
               center =  a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_ac[Ai] +  a_aw[Ai] + a_ae[Ai] +
                         a_bc[Ai] +  a_bw[Ai] +  a_be[Ai];
               left   = -a_cs[Ai] - a_csw[Ai] - a_cse[Ai]; /* front */
               right  = -a_cn[Ai] - a_cnw[Ai] - a_cne[Ai]; /* back */
               break;
         case 2: default:
               center =   a_cc[Ai] +  a_cw[Ai] +   a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] +
                          a_csw[Ai] + a_cse[Ai] +  a_cnw[Ai] - a_cne[Ai];
               left   =  -a_bc[Ai] -  a_bw[Ai] -   a_be[Ai]; /* below */
               right  =  -a_ac[Ai] -  a_aw[Ai] -   a_ae[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      hypre_BoxLoop2End(Ai, Pi);
   }
   else if (stencil_type15 == 1)
   {
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] +  a_as[Ai] + a_an[Ai] +
                         a_bc[Ai] + a_bs[Ai] + a_bn[Ai];
               left   = -a_cw[Ai] - a_csw[Ai] - a_cnw[Ai];
               right  = -a_ce[Ai] - a_cse[Ai] - a_cne[Ai];
               break;
            case 1:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] +  a_ac[Ai] +  a_bc[Ai];
               left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai]; /* front */
               right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai]; /* back */
               break;
         case 2: default:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] + a_cs[Ai] + a_cn[Ai] +
                         a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai];
               left   = -a_bc[Ai] - a_bs[Ai] - a_bn[Ai]; /* below */
               right  = -a_ac[Ai] - a_as[Ai] - a_an[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      hypre_BoxLoop2End(Ai, Pi);
   }
   else
   {
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] + a_as[Ai] + a_an[Ai] +
                         a_bc[Ai] + a_bs[Ai] + a_bn[Ai];
               left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai];
               right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai];
               break;
            case 1:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] +  a_ac[Ai] +  a_aw[Ai] + a_ae[Ai] +
                         a_bc[Ai] + a_bw[Ai] + a_be[Ai];
               left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai]; /* front */
               right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai]; /* back */
               break;
         case 2: default:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] + a_cs[Ai] + a_cn[Ai];
               left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai]; /* below */
               right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      hypre_BoxLoop2End(Ai, Pi);
   }
#undef DEVICE_VAR

   return hypre_error_flag;
}


HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS19
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(Pstenc1);
   HYPRE_UNUSED_VAR(rap_type);

   //hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   // hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   //HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   //HYPRE_Int              warning_cnt= 0;

   hypre_Index            index;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real           *a_csw, *a_cse, *a_cne, *a_cnw;
   HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real            *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, 0, 1);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 1);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 1);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 1);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, -1);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, -1);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, -1);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, -1);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_csw,a_cnw,a_ce,a_ae,a_be,a_cse,a_cne,Pp0,Pp1,p0,p1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai] + a_as[Ai] + a_an[Ai] + a_bs[Ai] +
                     a_bn[Ai];
            left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai];
            right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] +
                     a_be[Ai];
            left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai];
            right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai];
            break;
      case 2: default:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai]
                     + a_cne[Ai];
            left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai];
            right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai];
            break;
      };

      if (!center)
      {
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      //printf("Pp0[%d] = %e, Pp1 = %e\n",Pi,Pp0[Pi],Pp1[Pi]);
   }
   hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0_SS27
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
  hypre_Index        *P_stencil_shape )
{
   HYPRE_UNUSED_VAR(rap_type);
   HYPRE_UNUSED_VAR(Pstenc1);

   //hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   //hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   //HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   //HYPRE_Int              warning_cnt= 0;

   hypre_Index            index;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real           *a_csw, *a_cse, *a_cne, *a_cnw;
   HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_asw, *a_ase, *a_ane, *a_anw, *a_bsw, *a_bse, *a_bne, *a_bnw;
   HYPRE_Real            *p0, *p1;

   p0 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, 0, 1);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, 1);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, 1);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, 1);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 0, -1);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 0, -1);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, -1, -1);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 0, 1, -1);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, -1, 1);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, 1);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, 1);
   a_anw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, 1);
   a_ane = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, -1, -1);
   a_bsw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, -1, -1);
   a_bse = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, -1, 1, -1);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, i, index);

   hypre_SetIndex3(index, 1, 1, -1);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_csw,a_cnw,a_asw,a_anw,a_bsw,a_bnw,a_ce,a_ae,a_be,a_cse,a_cne,a_ase,a_ane,a_bse,a_bne,Pp0,Pp1,p0,p1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai] + a_as[Ai] + a_an[Ai] + a_bs[Ai] +
                     a_bn[Ai];
            left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai] - a_asw[Ai] - a_anw[Ai] - a_bsw[Ai]
                     - a_bnw[Ai];
            right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai] - a_ase[Ai] - a_ane[Ai] - a_bse[Ai]
                     - a_bne[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] +
                     a_be[Ai];
            left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai] - a_asw[Ai] - a_ase[Ai] - a_bsw[Ai]
                     - a_bse[Ai];
            right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai] - a_anw[Ai] - a_ane[Ai] - a_bnw[Ai]
                     - a_bne[Ai];
            break;
      case 2: default:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai]
                     + a_cne[Ai];
            left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai] - a_bsw[Ai] - a_bse[Ai] - a_bnw[Ai] -
                     a_bne[Ai];
            right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai] - a_asw[Ai] - a_ase[Ai] - a_anw[Ai] -
                     a_ane[Ai];
            break;
      };

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      //printf("Pp0[%d] = %e, Pp1 = %e\n",Pi,Pp0[Pi],Pp1[Pi]);
   }
   hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return hypre_error_flag;
}

#endif
