      SUBROUTINE ZLARTG( F, G, CS, SN, R )
*
*  -- LAPACK auxiliary routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     September 30, 1994
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   CS
      COMPLEX*16         F, G, R, SN
*     ..
*
*  Purpose
*  =======
*
*  ZLARTG generates a plane rotation so that
*
*     [  CS  SN  ]     [ F ]     [ R ]
*     [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
*     [ -SN  CS  ]     [ G ]     [ 0 ]
*
*  This is a faster version of the BLAS1 routine ZROTG, except for
*  the following differences:
*     F and G are unchanged on return.
*     If G=0, then CS=1 and SN=0.
*     If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any
*        floating point operations.
*
*  Arguments
*  =========
*
*  F       (input) COMPLEX*16
*          The first component of vector to be rotated.
*
*  G       (input) COMPLEX*16
*          The second component of vector to be rotated.
*
*  CS      (output) DOUBLE PRECISION
*          The cosine of the rotation.
*
*  SN      (output) COMPLEX*16
*          The sine of the rotation.
*
*  R       (output) COMPLEX*16
*          The nonzero component of the rotated vector.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
      COMPLEX*16         CZERO
      PARAMETER          ( CZERO = ( 0.0D+0, 0.0D+0 ) )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   D, DI, F1, F2, FA, G1, G2, GA
      COMPLEX*16         FS, GS, SS, T
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, DCONJG, DIMAG, SQRT
*     ..
*     .. Statement Functions ..
      DOUBLE PRECISION   ABS1, ABSSQ
*     ..
*     .. Statement Function definitions ..
      ABS1( T ) = ABS( DBLE( T ) ) + ABS( DIMAG( T ) )
      ABSSQ( T ) = DBLE( T )**2 + DIMAG( T )**2
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
         SN = DCONJG( G ) / ABS( G )
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
            SN = DCONJG( GS )*FS*( CS / F2 )
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
            SS = ( DCONJG( GS )*FS ) / ( FA*GA )
            SN = SS*DI
            R = G*SS*D
         END IF
      END IF
      RETURN
*
*     End of ZLARTG
*
      END
