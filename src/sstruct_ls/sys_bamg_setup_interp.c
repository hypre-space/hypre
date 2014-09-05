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
#include "hypre_lapack.h"

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

  //hypre_TFree( //P_Stencils ); // Cannot free this here!
  hypre_TFree(stencil_shape);

  return P;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPMatrixUnpack(
    const hypre_SStructPMatrix *M,
    const int                   nvars,
    hypre_StructMatrix       ***sM )
{
  int I, J;


  for ( I = 0; I < nvars; I++ ) {
    for ( J = 0; J < nvars; J++ ) {
      sM[I][J] = hypre_SStructPMatrixSMatrix(M,I,J);
    }
  }

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPVectorUnpack(
    const hypre_SStructPVector *V,
    const int                   nvars,
    hypre_StructVector        **sV )
{
  int I;

  for ( I = 0; I < nvars; I++ ) {
    sV[I] = hypre_SStructPVectorSVector(V,I);
  }

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructVectorUpdateGhostCells(
    hypre_StructVector  *sV,
    hypre_StructStencil *stencil )
{
  hypre_StructGrid    *grid = hypre_StructVectorGrid( sV );
  hypre_BoxArray      *dataspace = hypre_StructVectorDataSpace( sV );
  hypre_CommInfo      *info;
  hypre_CreateCommInfoFromStencil( grid, stencil, &info );
  hypre_CommPkg       *pkg;
  hypre_CommPkgCreate( info, dataspace, dataspace, 1, NULL, 0, hypre_StructGridComm(grid), &pkg );
  hypre_CommHandle    *handle;

  HYPRE_Complex *data = hypre_StructVectorData( sV );
  hypre_InitializeCommunication( pkg, data, data, 0, 0, &handle );
  hypre_FinalizeCommunication( handle );

  hypre_CommPkgDestroy( pkg );

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_CheckReturnValue( const char* func, HYPRE_Int rv )
{
  if ( rv != 0 ) {
    hypre_printf( "\nexit: error: %s returned %d\n", func, rv );
    exit(9);
  }
}


/*--------------------------------------------------------------------------
 * Use QR to determine P[i][j] - see http://www.netlib.org/lapack/lug/node40.html and
 * http://people.sc.fsu.edu/~jburkardt/f_src/qr_solve/qr_solve.html
 *
 * compute c_1 = Q^T b    (dim: (Mrows * Mrows) * (Mrows * 1) -> (Mrows * 1)),
 * compute x | R x = c_1  (dim: (Mcols*Mcols) (Mcols * 1) -> (Mcols * 1))
 *
 * M is Mrows by Mcols with Mrows > Mcols. Must be column-major order (Fortran style).
 * Q is Mrows by Mrows.
 *--------------------------------------------------------------------------*/

#if HYPRE_Complex == HYPRE_Real
#define hypre_xgeqrf hypre_dgeqrf
#define hypre_xxxmqr hypre_dormqr
#define hypre_xtrtrs hypre_dtrtrs
#else
#define hypre_xgeqrf hypre_zgeqrf
#define hypre_xxxmqr hypre_zunmqr
#define hypre_xtrtrs hypre_ztrtrs
#endif

HYPRE_Int hypre_LeastSquares(
    HYPRE_Complex* M,
    HYPRE_Int      Mrows,
    HYPRE_Int      Mcols,
    HYPRE_Complex* C,
    HYPRE_Int      Crows,
    HYPRE_Int      Ccols )
{

#if DEBUG_SYSBAMG > 1
  // print M and b to check
  hypre_printf("M | b for I=%d, iv=%d, k=%d\n", I, iv, k);
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) {
      hypre_printf("  %16.6e", M[Mi + Mj*Mrows]);
    }
    hypre_printf("  | %16.6e\n", C[Mi]);
  }
  hypre_printf("\n");
#endif

  HYPRE_Int       lwork = Mcols * 8;
  HYPRE_Complex*  work  = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, lwork);
  HYPRE_Complex*  tau   = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Mrows*Mcols);;
  HYPRE_Int       info;

  // NB: R and Q (via reflectors) are written to M
  hypre_xgeqrf( &Mrows, &Mcols, M, &Mrows, tau, work, &lwork, &info );
  hypre_CheckReturnValue( "hypre_xgeqrf", info );

#if DEBUG_SYSBAMG > 1
  // print Q\R to check
  hypre_printf("Q\\R for I=%d, iv=%d, k=%d\n", I, iv, k);
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) {
      hypre_printf("  %16.6e", M[Mi + Mj*Mrows]);
    }
    hypre_printf("\n");
  }
  hypre_printf("\n");
#endif

  // Q is Mrows x Mrows, 'M' = Mrows, 'N' = 1, 'K' = Mrows, 'A' = elementary reflector array = M
  hypre_xxxmqr( "Left", "Transpose", &Crows, &Ccols, &Mrows, M, &Mrows, tau, C, &Mrows, work, &lwork, &info );
  hypre_CheckReturnValue( "hypre_xxxmqr", info );

#if DEBUG_SYSBAMG > 1
  // print c to check
  hypre_printf("c for I=%d, iv=%d, k=%d\n", I, iv, k);
  for ( Mj = 0; Mj < Mcols; Mj++ ) {
    hypre_printf("  %16.6e\n", C[Mj]);
  }
  hypre_printf("\n");
#endif

  // Here, the matrix is R, which is Mcols by Mcols, upper triangular, and stored in M.
  hypre_xtrtrs( "Upper", "No transpose", "Non-unit", &Mcols, &Ccols, M, &Mrows, C, &Crows, &info );
  if ( info > 0 ) hypre_printf( "\nhypre_xtrtrs error: M is singular!" );
  hypre_CheckReturnValue( "hypre_xtrtrs", info );


#if DEBUG_SYSBAMG > 1
  // print x to check
  hypre_printf("x for I=%d, iv=%d, k=%d\n", I, iv, k);
  for ( Mj = 0; Mj < Mcols; Mj++ ) {
    hypre_printf("  %16.6e\n", C[Mj]);
  }
  hypre_printf("\n");
#endif

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
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

  HYPRE_Complex        *M, *C, *work, *tau;

  HYPRE_Int             ndims, Mrows, Mcols, Mi, Mj, Crows, Ccols, lwork, info;

  HYPRE_Int             b, I, J, i, j, k, sj, iP, iv;

  // XXX: NB: Assume same structure for all I,J; i.e., use [0][0] as representative.

  //  for each row, i = (I = i_vars, iP = i_grid),
  //    for each vector, k, and col, j = (J = j_vars, sj = j_grid), s.t. P[i][j] != 0,
  //      compute P[_i_][j] by minimizing l2norm( weight[k] * (v_c[k][j] P[_i_][j] - v_f[k][_i_]) )
  //                      by QR factorizing M[j][k] = v_c[k][j], etc.

  // P_Stencil dictates which P[i][j] != 0

  P_Stencil       = hypre_StructMatrixStencil(sP[0][0]);
  P_StencilShape = hypre_StructStencilShape(P_Stencil);
  P_StencilSize  = hypre_StructStencilSize(P_Stencil);

  ndims = hypre_StructStencilNDim(P_Stencil);

  hypre_SetIndex(stridec, 1);

  v_offsets = (HYPRE_Int*) hypre_TAlloc(HYPRE_Int, P_StencilSize);

  Mrows = nvecs;
  Mcols = nvars * P_StencilSize;
  M     = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Mrows*Mcols);

  Crows = Mrows;
  Ccols = 1;
  C     = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Crows*Ccols);

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

      for ( sj = 0; sj < P_StencilSize; sj++ )
        v_offsets[sj] = hypre_BoxOffsetDistance( vDataBox, P_StencilShape[sj] );

      hypre_BoxLoop2Begin( ndims, BoxSize, PDataBox, startc, stridec, iP, vDataBox, start, stride, iv );

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,iP,iv,J) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(iP, iv) {
        for ( k = 0; k < nvecs; k++ ) {
          Mi = k;

          C[Mi] = hypre_StructVectorData( sv[k][I] )[iv];

          for ( J = 0; J < nvars; J++ ) {
            for ( sj = 0; sj < P_StencilSize; sj++ ) {
              Mj = J*P_StencilSize + sj;
              M[Mi + Mj*Mrows] = hypre_StructVectorData( sv[k][J] )[iv + v_offsets[sj]];
            }
          }
        }

        hypre_LeastSquares( M, Mrows, Mcols, C, Crows, Ccols );

        for ( J = 0; J < nvars; J++ ) {
          for ( sj = 0; sj < P_StencilSize; sj++ ) {
            Mj = J*P_StencilSize + sj;
            hypre_StructMatrixBoxData(sP[I][J], b, sj)[iP] = C[Mj];   // 0.5 for sanity check
          }
        }
      }
      hypre_BoxLoop2End(iP, iv);

    }
  }

  for ( I = 0; I < nvars; I++ ) {
    for ( J = 0; J < nvars; J++ ) {
      hypre_StructInterpAssemble(sA[I][J], sP[I][J], 0, cdir, findex, stride);
    }
  }

  hypre_TFree( tau );
  hypre_TFree( work );
  hypre_TFree( v_offsets );
  hypre_TFree( C );
  hypre_TFree( M );

  return hypre_error_flag;
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
  HYPRE_Int              I,J,k;   // vars

  hypre_StructMatrix  ***sA;
  hypre_StructMatrix  ***sP;

  hypre_StructVector  ***sv;

  nvars = hypre_SStructPMatrixNVars(A);

  sA = hypre_TAlloc( hypre_StructMatrix**, nvars );
  sP = hypre_TAlloc( hypre_StructMatrix**, nvars );

  for ( I = 0; I < nvars; I++ ) {
    sA[I] = hypre_TAlloc( hypre_StructMatrix*, nvars );
    sP[I] = hypre_TAlloc( hypre_StructMatrix*, nvars );
  }

  hypre_SStructPMatrixUnpack( A, nvars, sA );
  hypre_SStructPMatrixUnpack( P, nvars, sP );

  sv = hypre_TAlloc( hypre_StructVector**, nvecs );

  for ( k = 0; k < nvecs; k++ ) {
    sv[k] = hypre_TAlloc( hypre_StructVector*, nvars );
    hypre_SStructPVectorUnpack( vecs[k], nvars, sv[k] );
  }

  // update vector ghost cells
  for ( k = 0; k < nvecs; k++ ) {
    for ( I = 0; I < nvars; I++ ) {
      hypre_StructVectorUpdateGhostCells( sv[k][I], hypre_StructMatrixStencil( sP[I][I] ) );
    }
  }

  hypre_SysBAMGSetupInterpOpLS(sA, sP, nvars, cdir, findex, stride, nvecs, sv);

  for ( I = 0; I < nvars; I++ ) {
    hypre_TFree( sA[I] );
    hypre_TFree( sP[I] );
  }

  hypre_TFree( sA );
  hypre_TFree( sP );

  for ( k = 0; k < nvecs; k++ ) {
    hypre_TFree( sv[k] );
  }

  hypre_TFree( sv );

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGComputeSVecs(
    hypre_SStructPMatrix*   A,
    HYPRE_Int               nsvecs,
    hypre_SStructPVector**  svecs )
{
  return hypre_error_flag;
}

