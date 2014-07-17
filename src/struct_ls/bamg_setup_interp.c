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

  HYPRE_Real            **Pp, **vp;
  HYPRE_Int              *v_offsets;

  HYPRE_Int              constant_coefficient;

  hypre_StructStencil   *P_stencil;
  hypre_Index           *P_stencil_shape;
  HYPRE_Int              P_stencil_size;

  hypre_Index            loop_size;
  hypre_Index            start;
  hypre_IndexRef         startc;
  hypre_Index            stridec;

  HYPRE_Int              i, j, k, Pi, vi, d;

  HYPRE_Real             smm, smp, smz, spp, spz, vkm, vkp, vkz;

  /*----------------------------------------------------------
   * Initialize some things
   *----------------------------------------------------------*/

  P_stencil       = hypre_StructMatrixStencil(P);
  P_stencil_shape = hypre_StructStencilShape(P_stencil);
  P_stencil_size  = hypre_StructStencilSize(P_stencil);

  // NB: P is accessed via a pointer for each stencil, but tv[k] via pointer + offsets (cf pfmg)

  Pp = (HYPRE_Real**) hypre_TAlloc(HYPRE_Real*, P_stencil_size);

  vp = (HYPRE_Real**) hypre_TAlloc(HYPRE_Real*, num_tv);
  
  v_offsets = (HYPRE_Int*) hypre_TAlloc(HYPRE_Int, P_stencil_size);

  hypre_SetIndex(stridec, 1);   // i.e. stride on coarse grid, i.e. 1,1,...

  /*----------------------------------------------------------
   * Compute P using analytical 2-pt LS solution
   *----------------------------------------------------------*/

  compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));

  hypre_ForBoxI(i, compute_boxes)
  {
    compute_box = hypre_BoxArrayBox(compute_boxes, i);

    startc = hypre_BoxIMin(compute_box);

    // 'start' : index of first F point for interpolation
    hypre_StructMapCoarseToFine(startc, findex, stride, start);
    
    hypre_BoxGetStrideSize(compute_box, stridec, loop_size);

    bamg_dbgmsg("findex:  %d %d %d\n", hypre_IndexD(findex, 0), hypre_IndexD(findex, 1), hypre_IndexD(findex, 2));
    bamg_dbgmsg("stride:  %d %d %d\n", hypre_IndexD(stride, 0), hypre_IndexD(stride, 1), hypre_IndexD(stride, 2));
    bamg_dbgmsg("stridec: %d %d %d\n", hypre_IndexD(stridec,0), hypre_IndexD(stridec,1), hypre_IndexD(stridec,2));
    bamg_dbgmsg("startc:  %d %d %d\n", hypre_IndexD(startc, 0), hypre_IndexD(startc, 1), hypre_IndexD(startc, 2));
    bamg_dbgmsg("start:   %d %d %d\n", hypre_IndexD(start,  0), hypre_IndexD(start,  1), hypre_IndexD(start,  2));

    P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

    for ( j = 0; j < P_stencil_size; j++ )
      Pp[j] = hypre_StructMatrixBoxData(P, i, j);

    bamg_dbgmsg("tv_dbox\n");

    tv_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(tv[0]),i);

    bamg_dbgmsg("vp[k]\n");

    for ( k = 0; k < num_tv; k++ )
      vp[k] = hypre_StructVectorBoxData(tv[k],i);
    
    bamg_dbgmsg("v_offsets[j]\n");

    for ( j = 0; j < P_stencil_size; j++ )
      v_offsets[j] = hypre_BoxOffsetDistance( tv_dbox, P_stencil_shape[j] );

    bamg_dbgmsg("v_offsets: %d %d\n", v_offsets[0], v_offsets[1]);

    /* No constant_coefficient switch */

    hypre_BoxLoop2Begin( hypre_StructMatrixNDim(P), loop_size,
                         P_dbox,  startc, stridec, Pi,
                         tv_dbox, start, stride, vi);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Pi,vi) HYPRE_SMP_SCHEDULE
#endif
    hypre_BoxLoop2For(Pi, vi)
    {
#if 0
      for ( j = 0; j < P_stencil_size; j++ )
        Pp[j][Pi] = 0.5;
#else
      smm = smp = smz = spp = spz = 0.0;

      for ( k = 0; k < num_tv; k++ )
      {
        vkm = vp[k][vi+v_offsets[0]];
        vkp = vp[k][vi+v_offsets[1]];
        vkz = vp[k][vi];
        smm += vkm * vkm;
        smp += vkm * vkp;
        smz += vkm * vkz;
        spp += vkp * vkp;
        spz += vkp * vkz;
        bamg_dbgmsg("vi: %d k: %d v{-,0,+}: %12.5e %12.5e %12.5e\n", vi, k, vkm, vkz, vkp);
      }
#if 1
      if ( spp == 0.0 ) {
        Pp[1][Pi] = 0.0;
      }
      else if ( smm == 0.0 ) {
        Pp[1][Pi] = 1.0;
      }
      else if ( fabs( 1 - smm/smp ) < 1e-8 || fabs( 1 - smm/smz ) < 1e-8 ) {
        Pp[1][Pi] = 0.0;
      }
      else {
        Pp[1][Pi] = ( smz - smm ) / (smp - smm);
        HYPRE_Real P_limit = 0.25;
        if ( Pp[1][Pi] < 0.50 - P_limit ) Pp[1][Pi] = 0.50 - P_limit;
        if ( Pp[1][Pi] > 0.50 + P_limit ) Pp[1][Pi] = 0.50 + P_limit;
      }
      Pp[0][Pi] = 1.0 - Pp[1][Pi];
#else
      Pp[0][Pi] = ( smz / smp - spz / spp ) / ( smm / smp - smp / spp );
      Pp[1][Pi] = ( spz / smp - smz / smm ) / ( spp / smp - smp / smm );
#endif
      bamg_dbgmsg("%12.5e %12.5e %12.5e %12.5e %12.5e . %12.5e %12.5e\n",
                   smm, smp, smz, spp, spz,        Pp[0][Pi], Pp[1][Pi]);
#endif
    }
    hypre_BoxLoop2End(Pi, vi);

  }

  hypre_StructInterpAssemble(A, P, 0, cdir, findex, stride);

  return hypre_error_flag;
}

