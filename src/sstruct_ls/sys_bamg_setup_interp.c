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

#include <HYPRE_config.h>   // for HYPRE_COMPLEX

#define hypre_re_im( x ) hypre_creal(x), hypre_cimag(x)

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_SStructPMatrix * hypre_SysBAMGCreateInterpOp
(
  hypre_SStructPMatrix*   A,
  hypre_SStructPGrid*     coarsePGrid,
  HYPRE_Int               cdir
)
{
  hypre_SStructPMatrix*   P;

  hypre_Index*            stencil_shape;
  HYPRE_Int               stencil_size;

  HYPRE_Int               NDim;

  HYPRE_Int               NVars;
  hypre_SStructStencil**  P_Stencils;

  HYPRE_Int               I,si;

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
  NDim  = hypre_StructStencilNDim(hypre_SStructPMatrixSStencil(A, 0, 0));
  NVars = hypre_SStructPMatrixNVars(A);
  P_Stencils = hypre_CTAlloc(hypre_SStructStencil *, NVars);

  for (I = 0; I < NVars; I++)
  {
    HYPRE_SStructStencilCreate(NDim, stencil_size, &P_Stencils[I]);

    // XXX: for inter-var interpolation, loop over I and J here, and set 0's with IJ matrix arg

    for (si = 0; si < stencil_size; si++)
      HYPRE_SStructStencilSetEntry(P_Stencils[I], si, stencil_shape[si], I);
  }

  /* create interpolation matrix */
  hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), coarsePGrid, P_Stencils, &P);

  //hypre_TFree( //P_Stencils ); // Cannot free this here!
  hypre_TFree(stencil_shape);

  return P;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPMatrixUnpack
(
  const hypre_SStructPMatrix* M,
  const int                   NVars,
  hypre_StructMatrix***       sM
)
{
  int I, J;

  for ( I = 0; I < NVars; I++ ) {
    for ( J = 0; J < NVars; J++ ) {
      sM[I][J] = hypre_SStructPMatrixSMatrix(M,I,J);
      //sysbamg_dbgmsg("sM[%d][%d] %p\n", I, J, sM[I][J]);
    }
  }

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPVectorUnpack
(
    const hypre_SStructPVector* V,
    const int                   NVars,
    hypre_StructVector**        sV
)
{
  int I;

  for ( I = 0; I < NVars; I++ ) {
    sV[I] = hypre_SStructPVectorSVector(V,I);
  }

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_StructVectorUpdateGhostCells
(
  hypre_StructVector*     sV,
  hypre_StructStencil*    stencil
)
{
  hypre_StructGrid*       SGrid             = hypre_StructVectorGrid( sV );
  hypre_BoxArray*         dataspace         = hypre_StructVectorDataSpace( sV );
  hypre_CommInfo*         info;

  hypre_CreateCommInfoFromStencil( SGrid, stencil, &info );

  hypre_CommPkg*          pkg;

  hypre_CommPkgCreate( info, dataspace, dataspace, 1, NULL, 0, hypre_StructGridComm(SGrid), &pkg );

  hypre_CommHandle*       handle;

  HYPRE_Complex*          data              = hypre_StructVectorData( sV );

  hypre_InitializeCommunication( pkg, data, data, 0, 0, &handle );
  hypre_FinalizeCommunication( handle );

  hypre_CommPkgDestroy( pkg );

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_CheckReturnValue
(
  const char*             func,
  HYPRE_Int               rv
)
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

#ifdef HYPRE_COMPLEX
#define hypre_xgeqrf hypre_zgeqrf
#define hypre_xxxmqr hypre_zunmqr
#define hypre_xtrtrs hypre_ztrtrs
#define TRANS "C"
#else
#define hypre_xgeqrf hypre_dgeqrf
#define hypre_xxxmqr hypre_dormqr
#define hypre_xtrtrs hypre_dtrtrs
#define TRANS "T"
#endif

HYPRE_Int hypre_LS
(
  HYPRE_Complex*          M,
  HYPRE_Int               Mrows,
  HYPRE_Int               Mcols,
  HYPRE_Complex*          C,
  HYPRE_Int               Crows,
  HYPRE_Int               Ccols
)
{
  //sysbamg_dbgmsg("M=%p Mrows=%d Mcols=%d C=%p Crows=%d Ccols=%d\n", M, Mrows, Mcols, C, Crows, Ccols);

  HYPRE_Int Mi, Mj;

#if DEBUG_SYSBAMG > 1
  // print M and b to check
  hypre_printf("hypre_LS: M | C = \n");
  for ( Mi = 0; Mi < Mrows; Mi++ )
  {
    for ( Mj = 0; Mj < Mcols; Mj++ )
      hypre_printf("  ( %16.6e %16.6e )", hypre_re_im(M[Mi+Mj*Mrows]));
    hypre_printf("  | ( %16.6e %16.6e )\n", hypre_re_im(C[Mi]));
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
  hypre_printf("hypre_LS: Q\\R = \n");
  for ( Mi = 0; Mi < Mrows; Mi++ )
  {
    for ( Mj = 0; Mj < Mcols; Mj++ ) hypre_printf("  ( %16.6e %16.6e )", hypre_re_im(M[Mi+Mj*Mrows]));
    hypre_printf("\n");
  }
  hypre_printf("\n");
#endif

  // Q is Mrows x Mrows, 'M' = Mrows, 'N' = 1, 'K' = Mrows, 'A' = elementary reflector array = M
  hypre_xxxmqr( "Left", TRANS, &Crows, &Ccols, &Mrows, M, &Mrows, tau, C, &Mrows, work, &lwork, &info );
  hypre_CheckReturnValue( "hypre_xxxmqr", info );

#if DEBUG_SYSBAMG > 1
  // print c to check
  hypre_printf("c\n");
  for ( Mj = 0; Mj < Mcols; Mj++ ) hypre_printf("  ( %16.6e %16.6e )", hypre_re_im(C[Mj]));
  hypre_printf("\n");
#endif

  // Here, the matrix is R, which is Mcols by Mcols, upper triangular, and stored in M.
  hypre_xtrtrs( "Upper", "No transpose", "Non-unit", &Mcols, &Ccols, M, &Mrows, C, &Crows, &info );
  if ( info  >  0 ) {
    hypre_printf( "XXX hypre_xtrtrs error: M is singular! Using C[*] = %g.\n", 1.0/Mcols);
    for ( Mj = 0; Mj < Mcols; Mj++ ) C[Mj] = 1.0/Mcols;     // XXX set to naive avg if trtrs fails
    exit(9);
  }
  else if ( info == -7 ) {
    hypre_printf( "\nhypre_xtrtrs error: the number of test vectors must be greater"
                  " than the stencil size. ( %d < %d )", Mrows, Mcols );
    exit(9);
  }
  else {
    hypre_CheckReturnValue( "hypre_xtrtrs", info );
  }

#if DEBUG_SYSBAMG > 1
  // print x to check
  hypre_printf("x\n");
  for ( Mj = 0; Mj < Mcols; Mj++ ) hypre_printf("  ( %16.6e %16.6e )", hypre_re_im(C[Mj]));
  hypre_printf("\n");
#endif

  hypre_TFree( tau );
  hypre_TFree( work );

  //printf("%s %3d %s finished.\n", __FILE__, __LINE__, __func__);

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGSetupInterpOpLS
(
  hypre_StructMatrix***   sA,
  hypre_StructMatrix***   sP,
  HYPRE_Int               NVars,
  HYPRE_Int               cdir,
  hypre_Index             findex,
  hypre_Index             stride,
  HYPRE_Int               nvecs,
  hypre_StructVector***   sv
)
{
  hypre_BoxArray*         GridBoxes;
  hypre_Box*              GridBox;

  hypre_StructStencil*    P_Stencil;
  hypre_Index*            P_StencilShape;
  HYPRE_Int               P_StencilSize;

  hypre_IndexRef          startc;
  hypre_Index             BoxSize, start, stridec;

  hypre_Box*              PDataBox, *vDataBox;

  HYPRE_Int*              v_offsets;

  HYPRE_Complex*          M, *C;

  HYPRE_Int               NDim, numIJ, Mrows, Mcols, Mi, Mj, Crows, Ccols;

  HYPRE_Int               b, I, J, i, j, k, sj, iP, iv;

  // XXX: NB: Assume same structure for all I,J; i.e., use [0][0] as representative.

  //  for each row, i = (I = i_vars, iP = i_grid),
  //    for each vector, k, and col, j = (J = j_vars, sj = j_grid), s.t. P[i][j] != 0,
  //      compute P[_i_][j] by minimizing l2norm( weight[k] * (v_c[k][j] P[_i_][j] - v_f[k][_i_]) )
  //                      by QR factorizing M[j][k] = v_c[k][j], etc.

  // P_Stencil dictates which P[i][j] != 0

  P_Stencil       = hypre_StructMatrixStencil(sP[0][0]);
  P_StencilShape  = hypre_StructStencilShape(P_Stencil);
  P_StencilSize   = hypre_StructStencilSize(P_Stencil);

  NDim = hypre_StructStencilNDim(P_Stencil);

  hypre_SetIndex(stridec, 1);

  v_offsets = (HYPRE_Int*) hypre_TAlloc(HYPRE_Int, P_StencilSize);

  for ( J = 0, numIJ = 0; J < NVars; J++ ) {
    if ( sP[0][J] != NULL ) numIJ++;    // XXX assume numIJ same for all I
  }

  Mrows = nvecs;
  Mcols = numIJ * P_StencilSize;

  M     = (HYPRE_Complex*) hypre_CTAlloc(HYPRE_Complex, Mrows*Mcols);

  Crows = Mrows;
  Ccols = 1;
  C     = (HYPRE_Complex*) hypre_CTAlloc(HYPRE_Complex, Crows*Ccols);

  //sysbamg_dbgmsg("Mrows %d  Mcols %d  Crows %d  Ccols %d\n", Mrows, Mcols, Crows, Ccols);

  GridBoxes = hypre_StructGridBoxes( hypre_StructMatrixGrid(sP[0][0]) );

  hypre_ForBoxI(b, GridBoxes)
  {
    GridBox = hypre_BoxArrayBox(GridBoxes, b);  // NB: GridBox is from P and corresponds to coarse grid

    startc = hypre_BoxIMin(GridBox);

    hypre_BoxGetStrideSize(GridBox, stridec, BoxSize);

    hypre_StructMapCoarseToFine(startc, findex, stride, start);

    for ( I = 0; I < NVars; I++ )
    {
      PDataBox = hypre_BoxArrayBox( hypre_StructMatrixDataSpace(sP[0][0]), b );
      vDataBox = hypre_BoxArrayBox( hypre_StructVectorDataSpace(sv[0][I]), b );

      for ( sj = 0; sj < P_StencilSize; sj++ )
        v_offsets[sj] = hypre_BoxOffsetDistance( vDataBox, P_StencilShape[sj] );

      hypre_BoxLoop2Begin( NDim, BoxSize, PDataBox, startc, stridec, iP, vDataBox, start, stride, iv );

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,iP,iv,J) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(iP, iv)
      {
#if DEBUG_SYSBAMG > 1
        hypre_Index iIndex;
        hypre_BoxLoopGetIndex( iIndex );
        sysbamg_dbgmsg("Set up LS - iP %d iv %d\n", iP, iv);
        printIndex( iIndex, NDim ); // dbgmsg
#endif

        for ( k = 0; k < nvecs; k++ )
        {
          Mi = k;

          C[Mi] = hypre_StructVectorBoxData( sv[k][I], b )[iv];

          for ( J = 0, numIJ = 0; J < NVars; J++ )
          {
            if ( sP[I][J] == NULL ) continue;

            for ( sj = 0; sj < P_StencilSize; sj++ ) {
              Mj = numIJ*P_StencilSize + sj;
              M[Mi + Mj*Mrows] = hypre_StructVectorBoxData( sv[k][J], b )[iv + v_offsets[sj]];
              //sysbamg_dbgmsg("I %d k %d J %d sj %d Mi %d Mj %d M ( %16.6e %16.6e )\n", I, k, J, sj, Mi, Mj, M[Mi+Mj*Mrows]);
            }

            numIJ++;
          }
        }

        hypre_LS( M, Mrows, Mcols, C, Crows, Ccols );

        for ( J = 0; J < NVars; J++ )
        {
          if ( sP[I][J] == NULL ) continue;

          for ( sj = 0; sj < P_StencilSize; sj++ ) {
            Mj = J*P_StencilSize + sj;
#if DEBUG_SYSBAMG_PFMG
            hypre_StructMatrixBoxData(sP[I][J], b, sj)[iP] = 0.5;     // to check against PFMG
#else
            hypre_StructMatrixBoxData(sP[I][J], b, sj)[iP] = C[Mj];
#endif
          }
        }
      }
      hypre_BoxLoop2End(iP, iv);
    }
  }

  printf("%s %3d %s\n", __FILE__, __LINE__, __func__);

  printf("findex\n"); printIndex(findex,NDim);
  printf("stride\n"); printIndex(stride,NDim);
  printf("cdir %d\n", cdir);

  for ( I = 0; I < NVars; I++ ) {
    for ( J = 0; J < NVars; J++ ) {
      if ( sP[I][J] == NULL ) continue;
      //sysbamg_dbgmsg("I %2d J %2d sA %p sP %p\n", I, J, sA[I][J], sP[I][J]);
      hypre_StructInterpAssemble(sA[I][J], sP[I][J], 0, cdir, findex, stride);
    }
  }

  sysbamg_dbgmsg("TFree\n");

  hypre_TFree( v_offsets );
  hypre_TFree( C );
  hypre_TFree( M );

  sysbamg_dbgmsg("return\n");

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGSetupInterpOp
(
  hypre_SStructPMatrix*   A,
  HYPRE_Int               cdir,
  hypre_Index             findex,
  hypre_Index             stride,
  hypre_SStructPMatrix*   P,
  HYPRE_Int               nvecs,
  hypre_SStructPVector**  vecs
)
{
  HYPRE_Int               NVars;
  HYPRE_Int               I,J,k;   // vars

  hypre_StructMatrix***   sA;
  hypre_StructMatrix***   sP;

  hypre_StructVector***   sv;

  NVars = hypre_SStructPMatrixNVars(A);

  sA = hypre_TAlloc( hypre_StructMatrix**, NVars );
  sP = hypre_TAlloc( hypre_StructMatrix**, NVars );

  for ( I = 0; I < NVars; I++ ) {
    sA[I] = hypre_TAlloc( hypre_StructMatrix*, NVars );
    sP[I] = hypre_TAlloc( hypre_StructMatrix*, NVars );
  }

  hypre_SStructPMatrixUnpack( A, NVars, sA );
  hypre_SStructPMatrixUnpack( P, NVars, sP );

  sv = hypre_TAlloc( hypre_StructVector**, nvecs );

  for ( k = 0; k < nvecs; k++ ) {
    sv[k] = hypre_TAlloc( hypre_StructVector*, NVars );
    hypre_SStructPVectorUnpack( vecs[k], NVars, sv[k] );
  }

  // update vector ghost cells
  for ( k = 0; k < nvecs; k++ ) {
    for ( I = 0; I < NVars; I++ ) {
      hypre_StructVectorUpdateGhostCells( sv[k][I], hypre_StructMatrixStencil( sP[I][I] ) );
    }
  }

  hypre_SysBAMGSetupInterpOpLS(sA, sP, NVars, cdir, findex, stride, nvecs, sv);

  for ( I = 0; I < NVars; I++ ) {
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



/*------------------------------------------------------------------------------
 * Compute the singular value decomposition of the coarse operator (single
 * process only at present) ala http://www.netlib.org/lapack/lug/node53.html.
 *
 * A is reduced to bidiagonal form: A = U_1 B V_1^T w/ U_1 and V_1 unitary
 * and B real and upper-bidiagonal for m >=n (lower bidiagonal for m < n).
 * NB: U_1 B V_1^T == Q B P^T.
 *
 * B is SV-decomposed: B = U_2 S V_2^T w/ U_2 and V_2 unitary and S diagonal.
 *
 * The L and R singular vectors of A are then the first min(Mrows,Mcols)
 * columns of U = U_1 U_2 and V = V_1 V_2.
 *
 * NB: xGBBRD (for banded matrices) may outperform xGEBRD for bidiagonalization.
 * NB: xBDSDC (for larger matrices) may outperform xBDSQR for the SVD.
 *
 * NB: given optimal blocksize, NB (8?), lwork >= { (M+N)*NB for xgebrd,
 *     4*N for xbdsqr, and max(M,N)*NB for xxxmbr }.
 *
 * NB: nsvecs must be no more than Mcols/2.
 *----------------------------------------------------------------------------*/

#ifdef HYPRE_COMPLEX
#define hypre_xgebrd hypre_zgebrd
#define hypre_xbdsqr hypre_zbdsqr
#define hypre_xxxmbr hypre_zunmbr
#else
#define hypre_xgebrd hypre_dgebrd
#define hypre_xbdsqr hypre_dbdsqr
#define hypre_xxxmbr hypre_dormbr
#endif

HYPRE_Int hypre_SVD
(
  HYPRE_Complex*          S,
  HYPRE_Complex*          M,
  HYPRE_Int               Mrows,
  HYPRE_Int               Mcols,
  HYPRE_Int               nsvecs,
  HYPRE_Int               symmetric
)
{
  HYPRE_Int               Mi, Mj;

#if DEBUG_SYSBAMG > 1
  // print M to check
  hypre_printf("hypre_SVD M:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) {
      hypre_printf("  %16.6e", M[Mi + Mj*Mrows]);
    }
    hypre_printf("\n");
  }
#endif

  HYPRE_Complex*  e     = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Mrows);
  HYPRE_Complex*  tauq  = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Mrows);
  HYPRE_Complex*  taup  = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, Mrows);
  HYPRE_Int       lwork = (Mrows + Mcols) * 8;  // optimal blocksize = 8?
  HYPRE_Complex*  work  = (HYPRE_Complex*) hypre_TAlloc(HYPRE_Complex, lwork);
  HYPRE_Int       info;

  // NB: R and Q (via reflectors) are written to M
  hypre_xgebrd( &Mrows, &Mcols, M, &Mrows, S, e, tauq, taup, work, &lwork, &info );
  hypre_CheckReturnValue( "hypre_xgebrd", info );

#if DEBUG_SYSBAMG > 1
  // print Q\R to check
  hypre_printf("hypre_SVD BD e:\n");
  hypre_printf("  %16s","");
  for ( Mi = 0; Mi < Mrows-1; Mi++ ) hypre_printf("  %16.6e", e[Mi]);
  hypre_printf("\n");
  hypre_printf("hypre_SVD BD d:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ )   hypre_printf("  %16.6e", S[Mi]);
  hypre_printf("\n");
#endif

  char            uplo = 'U';
  HYPRE_Int       zero = 0;
  HYPRE_Int       one  = 1;
  HYPRE_Complex*  U    = (HYPRE_Complex*) hypre_CTAlloc(HYPRE_Complex, Mrows*Mcols);
  HYPRE_Complex*  VT   = (HYPRE_Complex*) hypre_CTAlloc(HYPRE_Complex, Mrows*Mcols);

  // NB: xbdsqr : U -> U * Q and VT -> P^T V^T
  for ( Mi = 0; Mi < Mrows; Mi++ ) U[Mi + Mi*Mcols]  = 1.0;
  for ( Mi = 0; Mi < Mrows; Mi++ ) VT[Mi + Mi*Mcols] = 1.0;

  hypre_xbdsqr( &uplo, &Mrows, &Mrows, &Mrows, &zero, S, e, VT, &Mrows, U, &Mrows, NULL, &one, work, &info );
  hypre_CheckReturnValue( "hypre_xbdsqr", info );

#if DEBUG_SYSBAMG > 0
  hypre_printf("hypre_SVD S:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    hypre_printf("  %16.6e", S[Mi]);
  }
  hypre_printf("\n");
#endif

#if DEBUG_SYSBAMG > 1
  hypre_printf("hypre_SVD Q:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) hypre_printf("  %16.6e", U[Mi+Mj*Mrows]);
    hypre_printf("\n");
  }

  hypre_printf("hypre_SVD P^T:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) hypre_printf("  %16.6e", U[Mi+Mj*Mrows]);
    hypre_printf("\n");
  }

  hypre_printf("hypre_SVD [Q S P^T]:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) {
      HYPRE_Int     k;
      HYPRE_Complex x = 0.0;
      for ( k = 0; k < Mrows; k++ ) x += U[Mi+k*Mrows] * S[k] * VT[k+Mj*Mrows];
      hypre_printf("  %16.6e", ( fabs(x) < 1e-12 ? 0.0 : x ));
    }
    hypre_printf("\n");
  }
#endif

  // compute the singular vector matrices U = U_1 U_2 == Q U and V^T = V_2^T V_1^T == VT P^T

  char vect   = 'Q';
  char side   = 'L';
  char trans  = 'N';

  hypre_xxxmbr(&vect, &side, &trans, &Mrows, &Mcols, &Mcols, M, &Mcols, tauq, U, &Mcols, work, &lwork, &info);
  hypre_CheckReturnValue( "hypre_xxxmbr", info );

  vect   = 'P';
  side   = 'R';
  trans  = 'T';

  hypre_xxxmbr(&vect, &side, &trans, &Mcols, &Mrows, &Mcols, M, &Mcols, taup, VT, &Mrows, work, &lwork, &info);
  hypre_CheckReturnValue( "hypre_xxxmbr", info );

#if DEBUG_SYSBAMG > 1
  hypre_printf("hypre_SVD [U S V^T]:\n");
  for ( Mi = 0; Mi < Mrows; Mi++ ) {
    for ( Mj = 0; Mj < Mcols; Mj++ ) {
      HYPRE_Int     k;
      HYPRE_Complex x = 0.0;
      for ( k = 0; k < Mrows; k++ ) x += U[Mi+k*Mrows] * S[k] * VT[k+Mj*Mrows];
      hypre_printf("  %16.6e", ( fabs(x) < 1e-12 ? 0.0 : x ));
    }
    hypre_printf("\n");
  }
#endif

  // write lowest Mrows/2 L and R singular vectors into M := [v_l,1, v_l,2, ..., v_r,1, v_r,2, ...]
  //    nb: values/vectors are returned in* descending* order
  //    also reverse S to correspond to vector ordering

  for ( Mi = 0; Mi < Mrows; Mi++ )
  {
    if ( Mi < Mrows/2 ) {
      HYPRE_Complex tmp = S[Mi];
      S[Mi] = S[Mrows-1-Mi];
      S[Mrows-1-Mi] = tmp;
    }

    for ( Mj = 0; Mj < nsvecs; Mj++ )
      M[Mi+Mj*Mrows] = U[Mi+(Mcols-1-Mj)*Mrows];

    if ( ! symmetric )
      for ( Mj = 0; Mj < nsvecs; Mj++ )
        M[Mi+(nsvecs+Mj)*Mrows] = VT[(Mcols-1-Mj)+Mi*Mcols];
  }

  sysbamg_dbgmsg("%s finished\n", __func__);

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int IndexToInt
(
  const hypre_Index       Index,
  /*const*/ hypre_Box*    Box
)   // non const? XXX
{
  HYPRE_Int               Int, NDim, dim, stride;
  hypre_IndexRef          BoxMin, BoxMax;

  NDim    = hypre_BoxNDim( Box );
  BoxMin  = hypre_BoxIMin( Box );
  BoxMax  = hypre_BoxIMax( Box );

  Int = 0;
  stride = 1;

  for ( dim = 0; dim < NDim; dim++ ) {
    Int    += ( hypre_IndexD(Index,dim)  - hypre_IndexD(BoxMin,dim) ) * stride;
    stride *=   hypre_IndexD(BoxMax,dim) - hypre_IndexD(BoxMin,dim) + 1;
  }

  return Int;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int printIndex
(
  const hypre_Index       Index,
  const HYPRE_Int         NDim
)
{
  HYPRE_Int               dim;

  hypre_printf( "Index:" );

  for ( dim = 0; dim < NDim; dim++ ) {
    hypre_printf( "  %d", hypre_IndexD(Index,dim) );
  }

  hypre_printf( "\n" );

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int AddIndex(
  hypre_Index             Sum,
  const hypre_Index       A,
  const hypre_Index       B,
  /*const*/ hypre_Box*    Box
)
{
  HYPRE_Int               NDim, dim;
  hypre_Index             Size;

  NDim = hypre_BoxNDim( Box );
  hypre_BoxGetSize( Box, Size );

  for ( dim = 0; dim < NDim; dim++ ) {
    hypre_IndexD(Sum,dim) = hypre_IndexD(A,dim) + hypre_IndexD(B,dim);
    hypre_IndexD(Sum,dim) = (hypre_IndexD(Sum,dim) + hypre_IndexD(Size,dim)) % hypre_IndexD(Size,dim);
  }

  return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 * Compute singular vectors
 *  -- currently not at all parallel
 *
 *  NB: svecs must have space for 2*nsvecs and nsvecs must be no more than
 *      Mcols (see below)
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGComputeSVecs
(
  hypre_SStructPMatrix*   A,
  HYPRE_Int               nsvecs,
  hypre_SStructPVector**  svecs
)
{
  HYPRE_Int               NDim;
  HYPRE_Int               NVars;
  hypre_StructMatrix*     StructMatrix;
  hypre_StructVector*     StructVector;
  hypre_StructGrid*       Grid;
  hypre_BoxArray*         BoxArray;
  hypre_Box*              GridBox;
  hypre_Index             GridBoxSize;
  HYPRE_Int               GridBoxVolume;
  hypre_Box*              DataBox;
  hypre_Index             DataBoxSize;

  HYPRE_Int               BoxIdx = 0;   // XXX hard-wired* should loop over boxes* XXX
  HYPRE_Int               I, J, i, j, k, si, dim;
  hypre_IndexRef          start;
  hypre_Index             stride;
  hypre_Index             iIndex, jIndex;

  HYPRE_Complex*          M;
  HYPRE_Complex*          S;
  HYPRE_Int               Mrows, Mcols, Mi, Mj;

  hypre_StructStencil*    Stencil;
  hypre_Index*            StencilShape;
  HYPRE_Int               StencilSize;

  HYPRE_Int               symmetric;

  // get sizes and allocate M

  StructMatrix  = hypre_SStructPMatrixSMatrix( A, 0, 0 );  // XXX hard-wired XXX
  Grid          = hypre_StructMatrixGrid( StructMatrix );
  BoxArray      = hypre_StructGridBoxes( Grid );
  GridBox       = hypre_BoxArrayBox( BoxArray, BoxIdx );
  GridBoxVolume = hypre_BoxVolume( GridBox );

  hypre_BoxGetSize( GridBox, GridBoxSize );

  //sysbamg_dbgmsg( "GridBoxVolume = %d\n", GridBoxVolume );

  NVars = hypre_SStructPMatrixNVars( A );

  Mrows         = GridBoxVolume * NVars;
  Mcols         = Mrows;
  M             = (HYPRE_Complex*) hypre_CTAlloc( HYPRE_Complex, Mrows*Mcols );
  S             = (HYPRE_Complex*) hypre_CTAlloc( HYPRE_Complex, Mrows );

  // copy A into M

  NDim = hypre_SStructPMatrixNDim( A );

  sysbamg_dbgmsg( "Coarse Grid Min and Max:\n" )
  printIndex( hypre_BoxIMin( GridBox ), NDim ); // dbg
  printIndex( hypre_BoxIMax( GridBox ), NDim ); // dbg

  start         = hypre_BoxIMin( GridBox );
  hypre_SetIndex( stride, 1 );

  for ( I = 0; I < NVars; I++ )
  {
    for ( J = 0; J < NVars; J++ )
    {
      StructMatrix  = hypre_SStructPMatrixSMatrix( A, I, J );
      BoxArray      = hypre_StructMatrixDataSpace( StructMatrix );
      DataBox       = hypre_BoxArrayBox( BoxArray, BoxIdx );

      hypre_BoxGetSize( DataBox, DataBoxSize );

      Stencil       = hypre_StructMatrixStencil( StructMatrix );
      StencilSize   = hypre_StructStencilSize( Stencil );
      StencilShape  = hypre_StructStencilShape( Stencil );

      hypre_BoxLoop1Begin( NDim, GridBoxSize, DataBox, start, stride, i );

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,i,j,Mi,Mj) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For( i )
      {
        hypre_BoxLoopGetIndex( iIndex );  // note: relative to Min

        //sysbamg_dbgmsg( "iIndex:\n" );
        //printIndex( iIndex, NDim ); // dbg

        Mi = I * GridBoxVolume + IndexToInt( iIndex, GridBox );

        for ( si = 0; si < StencilSize; si++ )
        {
          AddIndex( jIndex, iIndex, StencilShape[si], GridBox );

          //sysbamg_dbgmsg( "StencilShape[%d] and jIndex:\n", si )
          //printIndex( StencilShape[si], NDim ); // dbg
          //printIndex( jIndex, NDim ); // dbg

          Mj = J * GridBoxVolume + IndexToInt( jIndex, GridBox );

          M[ Mi + Mj * Mrows ] = hypre_StructMatrixBoxData( StructMatrix, BoxIdx, si )[ i ];  // NB: column-major
          //sysbamg_dbgmsg( "Mi %3d Mj %3d M %12.3e I %d  J %d  i %d  si %d\n", Mi, Mj, M[Mi+Mj*Mrows], I, J, i, si );
        }
      }
      hypre_BoxLoop1End( i );
    }
  }

  symmetric     = hypre_SStructPMatrixSymmetric(A)[0][0];     // XXX assume var-indep symmetry

  // compute singular vectors

  hypre_SVD( S, M, Mrows, Mcols, nsvecs, symmetric );

  // copy lowest singular vectors into svecs
  //    M := [ v_l,1, v_l,2, ..., v_r,1, v_r,2, ... ] -> svecs[...]

  for ( k = 0; k < nsvecs * ( symmetric ? 1 : 2 ); k++ )
  {
    for ( I = 0; I < NVars; I++ )
    {
      StructVector  = hypre_SStructPVectorSVector( svecs[k], I );
      BoxArray      = hypre_StructVectorDataSpace( StructVector );
      DataBox       = hypre_BoxArrayBox( BoxArray, BoxIdx );

      hypre_BoxGetSize( DataBox, DataBoxSize );

      hypre_BoxLoop1Begin( NDim, GridBoxSize, DataBox, start, stride, i );

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,i,Mi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For( i )
      {
        hypre_BoxLoopGetIndex( iIndex );  // note: relative to Min

        //sysbamg_dbgmsg( "iIndex:\n" );
        //printIndex( iIndex, NDim ); // dbg

        Mi = I * GridBoxVolume + IndexToInt( iIndex, GridBox );

        hypre_StructVectorBoxData( StructVector, BoxIdx )[ i ] = M[ Mi + k * Mrows ]; // NB: column-major
      }
      hypre_BoxLoop1End( i );
    }
  }

#if DEBUG_SYSBAMG > 0
  // check that U^T[i,j] A[j,k] V[k,l] == U_i[j] A[j,k] V_l[k] = delta[i,l] S[i]
  HYPRE_Complex         x;
  hypre_SStructPVector* AV;
  hypre_SStructPGrid*   PGrid = hypre_SStructPVectorPGrid( svecs[0] );
  MPI_Comm              Comm  = hypre_SStructPGridComm( PGrid );

  hypre_SStructPVectorCreate( Comm, PGrid, &AV );
  hypre_SStructPVectorInitialize( AV );
  hypre_SStructPVectorAssemble( AV );

  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      hypre_SStructPMatvec( 1.0, A, svecs[j + (symmetric ? 0 : nsvecs)], 0.0, AV );
      hypre_SStructPComplexInnerProd( svecs[i], AV, &x );
      printf("SVD Check: U[k][%d] A[k,l] V[l][%d] / S[%d]: %16.6e\n", i, j, i, x / S[i]);
    }
  }

  hypre_SStructPVectorDestroy( AV );
#endif

  // clean up

  hypre_TFree( M );

  return hypre_error_flag;
}

