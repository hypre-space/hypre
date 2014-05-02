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
#include "bamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructMatrix* hypre_BAMGCreateInterpOp(
    hypre_StructMatrix *A,
    hypre_StructGrid   *cgrid,
    HYPRE_Int           cdir)
{
  hypre_StructMatrix   *P;

  hypre_StructStencil  *stencil;
  hypre_Index          *stencil_shape;
  HYPRE_Int             stencil_size;
  HYPRE_Int             stencil_dim;

  HYPRE_Int            *num_ghost;

  HYPRE_Int             i;

  /* set up stencil */
  stencil_size = 2;
  stencil_dim = hypre_StructStencilNDim(hypre_StructMatrixStencil(A));
  stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);

  hypre_SetIndex(stencil_shape[0], 0);
  hypre_IndexD(stencil_shape[0], cdir) = -1;

  hypre_SetIndex(stencil_shape[1], 0);
  hypre_IndexD(stencil_shape[1], cdir) =  1;
  
  stencil = hypre_StructStencilCreate(stencil_dim, stencil_size, stencil_shape);

  /* set up matrix */
  P = hypre_StructMatrixCreate(hypre_StructMatrixComm(A), cgrid, stencil);

  num_ghost = hypre_TAlloc(HYPRE_Int, stencil_dim);
  hypre_SetIndex(num_ghost, 1);
  hypre_StructMatrixSetNumGhost(P, num_ghost);

  /* constant_coefficient == 0 , all coefficients in A vary */
  hypre_StructMatrixSetConstantCoefficient( P, 0 );

  hypre_StructStencilDestroy(stencil);

  return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BAMGSetupInterpOp(
    hypre_StructMatrix   *A,
    HYPRE_Int             cdir,
    hypre_Index           findex,
    hypre_Index           stride,
    hypre_StructMatrix   *P,
    HYPRE_Int             num_tv,
    hypre_StructVector  **tv)
{
  hypre_BoxArray        *compute_boxes;
  hypre_Box             *compute_box;

  hypre_Box             *A_dbox;
  hypre_Box             *P_dbox;
  hypre_Box             *tv_dbox;

  HYPRE_Real            *Pp0, *Pp1, **tvp;
  HYPRE_Int              constant_coefficient;

  hypre_StructStencil   *P_stencil;
  hypre_Index           *P_stencil_shape;

  hypre_Index            loop_size;
  hypre_Index            start;
  hypre_IndexRef         startc;
  hypre_Index            stridec;

  HYPRE_Int              i, Pi, tvi, k;

  /*----------------------------------------------------------
   * Initialize some things
   *----------------------------------------------------------*/

  P_stencil       = hypre_StructMatrixStencil(P);
  P_stencil_shape = hypre_StructStencilShape(P_stencil);

  hypre_SetIndex(stridec, 1);

  tvp     = (HYPRE_Real**) hypre_TAlloc(HYPRE_Real*, num_tv);

  /*----------------------------------------------------------
   * Compute P
   *  - by analytically minimizing chi^2 for a 2-pt stencil
   *----------------------------------------------------------*/

  compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));

  hypre_ForBoxI(i, compute_boxes)
  {
    compute_box = hypre_BoxArrayBox(compute_boxes, i);

    startc  = hypre_BoxIMin(compute_box);
    hypre_StructMapCoarseToFine(startc, findex, stride, start);

    hypre_BoxGetStrideSize(compute_box, stridec, loop_size);

    P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

    Pp0 = hypre_StructMatrixBoxData(P, i, 0);
    Pp1 = hypre_StructMatrixBoxData(P, i, 1);

    tv_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(tv[0]),i);
    for ( k = 0; k < num_tv; k++ )
      tvp[k] = hypre_StructVectorBoxData(tv[k],i);

    /* No constant_coefficient switch */

    hypre_BoxLoop2Begin( hypre_StructMatrixNDim(P), loop_size,
                         P_dbox, startc, stridec, Pi,
                         tv_dbox, startc, stridec, tvi);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Ai,Pi,si,Ap) HYPRE_SMP_SCHEDULE
#endif
    hypre_BoxLoop2For(Pi, tvi)
    {
      Pp0[Pi] = 0.5;
      Pp1[Pi] = 0.5;
    }
    hypre_BoxLoop2End(Pi, tvi);

  }

  hypre_StructInterpAssemble(A, P, 0, cdir, findex, stride);

  return hypre_error_flag;
}

