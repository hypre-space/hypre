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

#include "_hypre_sstruct_ls.h"
#include "sys_bamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_SStructPMatrix * hypre_SysBAMGCreateInterpOp(
    hypre_SStructPMatrix *A,
    hypre_SStructPGrid   *cgrid,
    HYPRE_Int             cdir  )
{
  hypre_SStructPMatrix  *P;

  hypre_Index           *stencil_shape;
  HYPRE_Int              stencil_size;

  HYPRE_Int              ndims;

  HYPRE_Int              nvars;
  hypre_SStructStencil **P_Stencils;

  HYPRE_Int              I,si;

  /* set up stencil_shape */
  stencil_size = 2;
  stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
  for (si = 0; si < stencil_size; si++)
  {
    hypre_SetIndex(stencil_shape[si], 0);
  }
  hypre_IndexD(stencil_shape[0], cdir) = -1;
  hypre_IndexD(stencil_shape[1], cdir) =  1;

  /* set up P_Stencils */
  ndims = hypre_StructStencilNDim(hypre_SStructPMatrixSStencil(A, 0, 0));
  nvars = hypre_SStructPMatrixNVars(A);
  P_Stencils = hypre_CTAlloc(hypre_SStructStencil *, nvars);

  for (I = 0; I < nvars; I++)
  {
    HYPRE_SStructStencilCreate(ndims, stencil_size, &P_Stencils[I]);

    // XXX: for inter-var interpolation, loop over I and J here, and set 0's with IJ matrix arg

    for (si = 0; si < stencil_size; si++)
      HYPRE_SStructStencilSetEntry(P_Stencils[I], si, stencil_shape[si], I);
  }

  /* create interpolation matrix */
  hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), cgrid, P_Stencils, &P);

  hypre_TFree(stencil_shape);

  return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGSetupInterpOp(
    hypre_SStructPMatrix *A,
    HYPRE_Int             cdir,
    hypre_Index           findex,
    hypre_Index           stride,
    hypre_SStructPMatrix *P,
    HYPRE_Int             nvecs,
    hypre_SStructPVector **vecs )
{
  HYPRE_Int              nvars;
  HYPRE_Int              i,j,k;   // vars

  hypre_StructMatrix  ***sA;
  hypre_StructMatrix  ***sP;

  hypre_StructVector  ***sv;

  nvars = hypre_SStructPMatrixNVars(A);

  sA = hypre_TAlloc( hypre_StructMatrix**, nvars );
  sP = hypre_TAlloc( hypre_StructMatrix**, nvars );

  for ( i = 0; i < nvars; i++ ) {
    sA[i] = hypre_TAlloc( hypre_StructMatrix*, nvars );
    sP[i] = hypre_TAlloc( hypre_StructMatrix*, nvars );

    for ( j = 0; j < nvars; j++ ) {
      sA[i][j] = hypre_SStructPMatrixSMatrix(A,i,j);
      sP[i][j] = hypre_SStructPMatrixSMatrix(P,i,j);
    }
  }

  sv = hypre_TAlloc( hypre_StructVector**, nvecs );

  for ( k = 0; k < nvecs; k++ ) {
    sv[k] = hypre_TAlloc( hypre_StructVector*, nvars );

    for ( i = 0; i < nvars; i++ ) {
      sv[k][i] = hypre_SStructPVectorSVector( vecs[k], i );
    }
  }

  // update vector ghost cells
  {
    hypre_StructGrid    *tgrid = hypre_StructVectorGrid( sv[0][0] );
    hypre_BoxArray      *tdata = hypre_StructVectorDataSpace( sv[0][0] );
    hypre_StructStencil *tstencil = hypre_StructMatrixStencil( sP[0][0] );
    hypre_CommInfo      *tinfo;
    hypre_CreateCommInfoFromStencil( tgrid, tstencil, &tinfo );
    hypre_CommPkg       *tpkg;
    hypre_CommPkgCreate( tinfo, tdata, tdata, 1, NULL, 0, hypre_StructGridComm(tgrid), &tpkg );
    hypre_CommHandle    *thandle;

    for ( k = 0; k < nvecs; k++ ) {
      for ( i = 0; i < nvars; i++ ) {
        HYPRE_Complex *p = hypre_StructVectorData( sv[k][i] );
        hypre_InitializeCommunication( tpkg, p, p, 0, 0, &thandle );
        hypre_FinalizeCommunication( thandle );
      }
    }

    hypre_CommPkgDestroy( tpkg );
  }

  hypre_SysBAMGSetupInterpOpLS(sA, sP, nvars, cdir, findex, stride, nvecs, sv);

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 * set P[i][j] to minimize sum_k ( weight_k * || P[i][j] v_c[k][j] - v_f[k][i] ||^2 )
 * i := (I',i') := ((color,spin),x)
 * no weight_k for now
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGSetupInterpOpLS(
    hypre_StructMatrix ***sA,
    hypre_StructMatrix ***sP,
    HYPRE_Int             nvars,
    HYPRE_Int             cdir,
    hypre_Index           findex,
    hypre_Index           stride,
    HYPRE_Int             nvecs,
    hypre_StructVector ***sv )
{
  hypre_BoxArray       *GridBoxes;
  hypre_Box            *GridBox;

  hypre_StructStencil  *P_Stencil;
  hypre_Index          *P_StencilShape;
  HYPRE_Int             P_StencilSize;

  hypre_IndexRef        startc;
  hypre_Index           BoxSize, start, stridec;

  hypre_Box            *PDataBox, *vDataBox;

  HYPRE_Int            *v_offsets;

  HYPRE_Complex        *M;

  HYPRE_Int             ndims;

  HYPRE_Int             b, I, J, i, j, k, si, iP, iv;

  // XXX: NB: Assume structure of all vars is the same -> use [0][0] as representative.

  //  for each row, i = (i_var,i_grid), 
  //    for each vector, k, and col, j = (j_var,j_grid), s.t. P[i][j] != 0,
  //      compute P[i][j] by miminizing sum_k ( weight[k] * | P[i][j] v_c[j] - v_f[i] |^2 )
  //                      by QR factorizing M[j][k] = v[j][k], etc.

  // P_Stencil dictates which P[i][j] != 0

  P_Stencil       = hypre_StructMatrixStencil(sP[0][0]);
  P_StencilShape = hypre_StructStencilShape(P_Stencil);
  P_StencilSize  = hypre_StructStencilSize(P_Stencil);

  ndims = hypre_StructStencilNDim(P_Stencil);

  hypre_SetIndex(stridec, 1);

  v_offsets = (HYPRE_Int*) hypre_TAlloc(HYPRE_Int, P_StencilSize);

  M         = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, nvars*nvecs*P_StencilSize);

  GridBoxes = hypre_StructGridBoxes( hypre_StructMatrixGrid(sP[0][0]) );

  hypre_ForBoxI(b, GridBoxes)
  {
    GridBox = hypre_BoxArrayBox(GridBoxes, b);  // NB: GridBox is from P and corresponds to coarse grid

    startc = hypre_BoxIMin(GridBox);

    hypre_BoxGetStrideSize(GridBox, stridec, BoxSize);

    hypre_StructMapCoarseToFine(startc, findex, stride, start);

    for ( I = 0; I < nvars; I++ )
    {
      PDataBox = hypre_BoxArrayBox( hypre_StructMatrixDataSpace(sP[0][0]), b );
      vDataBox = hypre_BoxArrayBox( hypre_StructVectorDataSpace(sv[0][I]), b );

      for ( si = 0; si < P_StencilSize; si++ )
        v_offsets[si] = hypre_BoxOffsetDistance( vDataBox, P_StencilShape[si] );

      hypre_BoxLoop2Begin( ndims, BoxSize, PDataBox, startc, stridec, iP, vDataBox, start, stride, iv );

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,iP,iv,J) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(iP, iv)
      {
        for ( J = 0; J < nvars; J++ )
        {
          for ( si = 0; si < P_StencilSize; si++ )
          {
            for ( k = 0; k < nvecs; k++ )
            {
              M[(J*P_StencilSize + j)*nvecs + k] = hypre_StructVectorData( sv[k][J] )[iv + v_offsets[si]];
              if ( k == 0 )
                sysbamg_dbgmsg( "I = %d iv = %d J = %d si = %d k = %d M[k][j] = %f\n",
                                I, iv, J, si, k, hypre_StructVectorData( sv[k][J] )[iv + v_offsets[si]] );
            }
          }
        }

        // Use QR to determine P[i][j] - see http://www.netlib.org/lapack/lug/node40.html

        for ( J = 0; J < nvars; J++ )
        {
          for ( si = 0; si < P_StencilSize; si++ )
          {
            hypre_StructMatrixBoxData(sP[I][J], b, si)[iP] = 0.5;
          }
        }
      }
      hypre_BoxLoop2End(iP, iv);

    }
  }

  for ( I = 0; I < nvars; I++ )
  {
    for ( J = 0; J < nvars; J++ )
    {
      hypre_StructInterpAssemble(sA[I][J], sP[I][J], 0, cdir, findex, stride);
    }
  }

  hypre_TFree( v_offsets );
  hypre_TFree( M );

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/



