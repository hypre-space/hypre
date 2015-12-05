/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.18 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_PFMGCreateInterpOp
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
   stencil_dim = hypre_StructStencilDim(hypre_StructMatrixStencil(A));
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;
   stencil =
      hypre_StructStencilCreate(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   P = hypre_StructMatrixCreate(hypre_StructMatrixComm(A), cgrid, stencil);
   hypre_StructMatrixSetNumGhost(P, num_ghost);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient==2 )
   {
      if ( rap_type==0 )
      /* A has variable diagonal, which will force all P coefficients to be variable */
         hypre_StructMatrixSetConstantCoefficient(P, 0 );
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
 * hypre_PFMGSetupInterpOp
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
                        
   double                *Pp0, *Pp1;
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

   for (si = 0; si < stencil_size; si++)
   {
      mrk0 = 0;
      mrk1 = 0;
      for (d = 0; d < hypre_StructStencilDim(stencil); d++)
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
      if (mrk0 == hypre_StructStencilDim(stencil))
      {
         si0 = si;
      }
      if (mrk1 == hypre_StructStencilDim(stencil))
      {
         si1 = si;
      }
   }
            
   hypre_SetIndex(stridec, 1, 1, 1);

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

         if ( constant_coefficient==1 )
            /* all coefficients are constant */
         {
            hypre_PFMGSetupInterpOp_CC1
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
         }

         else if ( constant_coefficient==2 )
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
  double             *Pp0,
  double             *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   HYPRE_Int              si;
   HYPRE_Int              Ai, Pi;
   double                *Ap;
   double                 center;
   HYPRE_Int              Astenc;
   HYPRE_Int              loopi, loopj, loopk;
   HYPRE_Int              mrk0, mrk1;
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int              warning_cnt= 0;

   hypre_BoxLoop2Begin(loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,Pi,si,center,Ap,Astenc,mrk0,mrk1
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop2For(loopi, loopj, loopk, Ai, Pi)
      {
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
               mrk0++;
            if (si == si1 && Ap[Ai] == 0.0)
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
            Pp0[Pi] = 0.0;
         if (mrk1 != 0)
            Pp1[Pi] = 0.0;
      }
   hypre_BoxLoop2End(Ai, Pi);

   if (warning_cnt)
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return hypre_error_flag;
}

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
  double             *Pp0,
  double             *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   HYPRE_Int              si;
   HYPRE_Int              Ai, Pi;
   double                *Ap;
   double                 center;
   HYPRE_Int              Astenc;
   HYPRE_Int              mrk0, mrk1;
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int              warning_cnt= 0;

   Ai = hypre_CCBoxIndexRank(A_dbox,start );
   Pi = hypre_CCBoxIndexRank(P_dbox,startc);

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
         mrk0++;
      if (si == si1 && Ap[Ai] == 0.0)
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
      Pp0[Pi] = 0.0;
   if (mrk1 != 0)
      Pp1[Pi] = 0.0;

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
  double             *Pp0,
  double             *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   HYPRE_Int              si;
   HYPRE_Int              Ai;
   HYPRE_Int              Pi;
   double                *Ap;
   double                 P0, P1;
   double                 center, center_offd;
   HYPRE_Int              Astenc;
   HYPRE_Int              loopi, loopj, loopk;
   HYPRE_Int              mrk0, mrk1, mrk0_offd, mrk1_offd;
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   hypre_Index            diag_index;
   HYPRE_Int              diag_rank;
   HYPRE_Int              warning_cnt= 0;

   hypre_SetIndex(diag_index, 0, 0, 0);
   diag_rank = hypre_StructStencilElementRank(stencil, diag_index);

   if ( rap_type!=0 )
   {
      /* simply force P to be constant coefficient, all 0.5's */
      Pi = hypre_CCBoxIndexRank(P_dbox,startc);
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
      Ai = hypre_CCBoxIndexRank(A_dbox,start );

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
               mrk0_offd++;
            if (si == si1 && Ap[Ai] == 0.0)
               mrk1_offd++;
         }
      }

      si = diag_rank;
      hypre_BoxLoop2Begin(loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,Pi,center,Ap,Astenc,mrk0,mrk1
#include "hypre_box_smp_forloop.h"
      hypre_BoxLoop2For(loopi, loopj, loopk, Ai, Pi)
         {
            Pp0[Pi] = P0;
            Pp1[Pi] = P1;
            center = center_offd;
            mrk0 = mrk0_offd;
            mrk1 = mrk1_offd;

            Ap = hypre_StructMatrixBoxData(A, i, si);
            Astenc = hypre_IndexD(stencil_shape[si], cdir);
            hypre_assert( Astenc==0 );
            center += Ap[Ai];

            if (si == si0 && Ap[Ai] == 0.0)
               mrk0++;
            if (si == si1 && Ap[Ai] == 0.0)
               mrk1++;

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
               Pp0[Pi] = 0.0;
            if (mrk1 != 0)
               Pp1[Pi] = 0.0;

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
