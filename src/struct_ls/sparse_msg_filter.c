/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_struct_ls.h"

#if 0

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGFilterSetup( hypre_StructMatrix *A,
                            HYPRE_Int          *num_grids,
                            HYPRE_Int           lx,
                            HYPRE_Int           ly,
                            HYPRE_Int           lz,
                            HYPRE_Int           jump,
                            hypre_StructVector *visitx,
                            hypre_StructVector *visity,
                            hypre_StructVector *visitz    )
{
   HYPRE_Int             ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_dbox;
   hypre_Box             *v_dbox;
                        
   HYPRE_Int              Ai;
   HYPRE_Int              vi;
                        
   HYPRE_Real            *Ap;
   HYPRE_Real            *vxp;
   HYPRE_Real            *vyp;
   HYPRE_Real            *vzp;
   HYPRE_Real             lambdax;
   HYPRE_Real             lambday;
   HYPRE_Real             lambdaz;
   HYPRE_Real             lambda_max;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
                        
   HYPRE_Int              Astenc;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   HYPRE_Int              i, si, dir, k, l;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      hypre_SetIndex3(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      hypre_SetIndex3(stridev, 1, 1, 1);

      hypre_StructVectorSetConstantValues(visitx, 0.0);
      hypre_StructVectorSetConstantValues(visity, 0.0);
      hypre_StructVectorSetConstantValues(visitz, 0.0);
   }

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visitx), i);

      vxp = hypre_StructVectorBoxData(visitx, i);
      vyp = hypre_StructVectorBoxData(visity, i);
      vzp = hypre_StructVectorBoxData(visitz, i);

      start = hypre_BoxIMin(compute_box);
      hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      hypre_BoxGetSize(compute_box, loop_size);

      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start,  stride,  Ai,
                          v_dbox, startv, stridev, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,vi,lambdax,lambday,lambdaz,si,Ap,Astenc,lambda_max,dir) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(Ai, vi)
      {
         lambdax = 0.0;
         lambday = 0.0;
         lambdaz = 0.0;

         for (si = 0; si < stencil_size; si++)
         {
            Ap = hypre_StructMatrixBoxData(A, i, si);

            /* compute lambdax */
            Astenc = hypre_IndexD(stencil_shape[si], 0);
            if (Astenc == 0)
            {
               lambdax += Ap[Ai];
            }
            else
            {
               lambdax -= Ap[Ai];
            }

            /* compute lambday */
            Astenc = hypre_IndexD(stencil_shape[si], 1);
            if (Astenc == 0)
            {
               lambday += Ap[Ai];
            }
            else
            {
               lambday -= Ap[Ai];
            }

            /* compute lambdaz */
            Astenc = hypre_IndexD(stencil_shape[si], 2);
            if (Astenc == 0)
            {
               lambdaz += Ap[Ai];
            }
            else
            {
               lambdaz -= Ap[Ai];
            }
         }

         lambdax *= lambdax;
         lambday *= lambday;
         lambdaz *= lambdaz;

         lambda_max = 0;
         dir = -1;
         if ((lx < num_grids[0] - 1) && (lambdax > lambda_max))
         {
            lambda_max = lambdax;
            dir = 0;
         }
         if ((ly < num_grids[1] - 1) && (lambday > lambda_max))
         {
            lambda_max = lambday;
            dir = 1;
         }
         if ((lz < num_grids[2] - 1) && (lambdaz > lambda_max))
         {
            lambda_max = lambdaz;
            dir = 2;
         }

         if (dir == 0)
         {
            vxp[vi] = (HYPRE_Real) ( ((HYPRE_Int) vxp[vi]) | k );
         }
         else if (dir == 1)
         {
            vyp[vi] = (HYPRE_Real) ( ((HYPRE_Int) vyp[vi]) | k );
         }
         else if (dir == 2)
         {
            vzp[vi] = (HYPRE_Real) ( ((HYPRE_Int) vzp[vi]) | k );
         }
      }
      hypre_BoxLoop2End(Ai, vi);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGFilter( hypre_StructVector *visit,
                       hypre_StructVector *e,
                       HYPRE_Int           lx,
                       HYPRE_Int           ly,
                       HYPRE_Int           lz,
                       HYPRE_Int           jump  )
{
   HYPRE_Int             ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *e_dbox;
   hypre_Box             *v_dbox;
                        
   HYPRE_Int              ei;
   HYPRE_Int              vi;
                        
   HYPRE_Real            *ep;
   HYPRE_Real            *vp;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   HYPRE_Int              i, k, l;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      hypre_SetIndex3(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      hypre_SetIndex3(stridev, 1, 1, 1);
   }

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(e));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);

      e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
      v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visit), i);

      ep = hypre_StructVectorBoxData(e, i);
      vp = hypre_StructVectorBoxData(visit, i);

      start = hypre_BoxIMin(compute_box);
      hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      hypre_BoxGetSize(compute_box, loop_size);

      hypre_BoxLoop2Begin(hypre_StructVectorNDim(e), loop_size,
                          e_dbox, start,  stride,  ei,
                          v_dbox, startv, stridev, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,ei,vi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(ei, vi)
      {
         if ( !(((HYPRE_Int) vp[vi]) & k) )
         {
            ep[ei] = 0.0;
         }
      }
      hypre_BoxLoop2End(ei, vi);
   }

   return ierr;
}

#else

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGFilterSetup( hypre_StructMatrix *A,
                            HYPRE_Int          *num_grids,
                            HYPRE_Int           lx,
                            HYPRE_Int           ly,
                            HYPRE_Int           lz,
                            HYPRE_Int           jump,
                            hypre_StructVector *visitx,
                            hypre_StructVector *visity,
                            hypre_StructVector *visitz    )
{
   HYPRE_Int             ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_dbox;
   hypre_Box             *v_dbox;
                        
   HYPRE_Int              Ai;
   HYPRE_Int              vi;
                        
   HYPRE_Real            *Ap;
   HYPRE_Real            *vxp;
   HYPRE_Real            *vyp;
   HYPRE_Real            *vzp;
   HYPRE_Real             lambdax;
   HYPRE_Real             lambday;
   HYPRE_Real             lambdaz;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
                        
   HYPRE_Int              Astenc;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   HYPRE_Int              i, si;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visitx), i);

      vxp = hypre_StructVectorBoxData(visitx, i);
      vyp = hypre_StructVectorBoxData(visity, i);
      vzp = hypre_StructVectorBoxData(visitz, i);

      start = hypre_BoxIMin(compute_box);
      hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      hypre_BoxGetSize(compute_box, loop_size);

      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start,  stride,  Ai,
                          v_dbox, startv, stridev, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,vi,lambdax,lambday,lambdaz,si,Ap,Astenc) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(Ai, vi)
      {
         lambdax = 0.0;
         lambday = 0.0;
         lambdaz = 0.0;

         for (si = 0; si < stencil_size; si++)
         {
            Ap = hypre_StructMatrixBoxData(A, i, si);

            /* compute lambdax */
            Astenc = hypre_IndexD(stencil_shape[si], 0);
            if (Astenc == 0)
            {
               lambdax += Ap[Ai];
            }
            else
            {
               lambdax -= Ap[Ai];
            }

            /* compute lambday */
            Astenc = hypre_IndexD(stencil_shape[si], 1);
            if (Astenc == 0)
            {
               lambday += Ap[Ai];
            }
            else
            {
               lambday -= Ap[Ai];
            }

            /* compute lambdaz */
            Astenc = hypre_IndexD(stencil_shape[si], 2);
            if (Astenc == 0)
            {
               lambdaz += Ap[Ai];
            }
            else
            {
               lambdaz -= Ap[Ai];
            }
         }

         lambdax *= lambdax;
         lambday *= lambday;
         lambdaz *= lambdaz;

         vxp[vi] = lambdax / (lambdax + lambday + lambdaz);
         vyp[vi] = lambday / (lambdax + lambday + lambdaz);
         vzp[vi] = lambdaz / (lambdax + lambday + lambdaz);
      }
      hypre_BoxLoop2End(Ai, vi);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGFilter( hypre_StructVector *visit,
                       hypre_StructVector *e,
                       HYPRE_Int           lx,
                       HYPRE_Int           ly,
                       HYPRE_Int           lz,
                       HYPRE_Int           jump  )
{
   HYPRE_Int             ierr = 0;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *e_dbox;
   hypre_Box             *v_dbox;
                        
   HYPRE_Int              ei;
   HYPRE_Int              vi;
                        
   HYPRE_Real            *ep;
   HYPRE_Real            *vp;
                        
   hypre_Index            loop_size;
   hypre_Index            cindex;
   hypre_IndexRef         start;
   hypre_Index            startv;
   hypre_Index            stride;
   hypre_Index            stridev;
                        
   HYPRE_Int              i;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(e));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);

      e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e), i);
      v_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(visit), i);

      ep = hypre_StructVectorBoxData(e, i);
      vp = hypre_StructVectorBoxData(visit, i);

      start = hypre_BoxIMin(compute_box);
      hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      hypre_BoxGetSize(compute_box, loop_size);

      hypre_BoxLoop2Begin(hypre_StructVectorNDim(e), loop_size,
                          e_dbox, start,  stride,  ei,
                          v_dbox, startv, stridev, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,ei,vi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(ei, vi)
      {
         ep[ei] *= vp[vi];
      }
      hypre_BoxLoop2End(ei, vi);
   }

   return ierr;
}

#endif
