      SUBROUTINE CLARTG( F, G, CS, SN, R )
*
*  -- LAPACK auxiliary routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     September 30, 1994
*
*     .. Scalar Arguments ..
      REAL               CS
      COMPLEX            F, G, R, SN
*     ..
*
*  Purpose
*  =======
*
*  CLARTG generates a plane rotation so that
*
*     [  CS  SN  ]     [ F ]     [ R ]
*     [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
*     [ -SN  CS  ]     [ G ]     [ 0 ]
*
*  This is a faster version of the BLAS1 routine CROTG, except for
*  the following differences:
*     F and G are unchanged on return.
*     If G=0, then CS=1 and SN=0.
*     If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any
*        floating point operations.
*
*  Arguments
*  =========
*
*  F       (input) COMPLEX
*          The first component of vector to be rotated.
*
*  G       (input) COMPLEX
*          The second component of vector to be rotated.
*
*  CS      (output) REAL
*          The cosine of the rotation.
*
*  SN      (output) COMPLEX
*          The sine of the rotation.
*
*  R       (output) COMPLEX
*          The nonzero component of the rotated vector.
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ONE, ZERO
      PARAMETER          ( ONE = 1.0E+0, ZERO = 0.0E+0 )
      COMPLEX            CZERO
      PARAMETER          ( CZERO = ( 0.0E+0, 0.0E+0 ) )
*     ..
*     .. Local Scalars ..
      REAL               D, DI, F1, F2, FA, G1, G2, GA
      COMPLEX            FS, GS, SS, T
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, AIMAG, CONJG, REAL, SQRT
*     ..
*     .. Statement Functions ..
      REAL               ABS1, ABSSQ
*     ..
*     .. Statement Function definitions ..
      ABS1( T ) = ABS( REAL( T ) ) + ABS( AIMAG( T ) )
      ABSSQ( T ) = REAL( T )**2 + AIMAG( T )**2
*     ..
*     .. Executable Statements ..
*
*     [ 25 or 38 ops for main paths ]
*
      IF( G.EQ.CZERO ) THEN
         CS = ONE
         SN = ZERO
         R = F
      ELSE IF( F.EQ.CZERO ) THEN
         CS = ZERO
*
         SN = CONJG( G ) / ABS( G )
         R = ABS( G )
*
*         SN = ONE
*         R = G
*
      ELSE
         F1 = ABS1( F )
         G1 = ABS1( G )
         IF( F1.GE.G1 ) THEN
            GS = G / F1
            G2 = ABSSQ( GS )
            FS = F / F1
            F2 = ABSSQ( FS )
            D = SQRT( ONE+G2 / F2 )
            CS = ONE / D
            SN = CONJG( GS )*FS*( CS / F2 )
            R = F*D
         ELSE
            FS = F / G1
            F2 = ABSSQ( FS )
            FA = SQRT( F2 )
            GS = G / G1
            G2 = ABSSQ( GS )
            GA = SQRT( G2 )
            D = SQRT( ONE+F2 / G2 )
            DI = ONE / D
            CS = ( FA / GA )*DI
            SS = ( CONJG( GS )*FS ) / ( FA*GA )
            SN = SS*DI
            R = G*SS*D
         END IF
      END IF
      RETURN
*
*     End of CLARTG
*
      END
