      SUBROUTINE ZGTTRF( N, DL, D, DU, DU2, IPIV, INFO )
*
*  -- LAPACK routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     September 30, 1994
*
*     .. Scalar Arguments ..
      INTEGER            INFO, N
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      COMPLEX*16         D( * ), DL( * ), DU( * ), DU2( * )
*     ..
*
*  Purpose
*  =======
*
*  ZGTTRF computes an LU factorization of a complex tridiagonal matrix A
*  using elimination with partial pivoting and row interchanges.
*
*  The factorization has the form
*     A = L * U
*  where L is a product of permutation and unit lower bidiagonal
*  matrices and U is upper triangular with nonzeros in only the main
*  diagonal and first two superdiagonals.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  DL      (input/output) COMPLEX*16 array, dimension (N-1)
*          On entry, DL must contain the (n-1) subdiagonal elements of
*          A.
*          On exit, DL is overwritten by the (n-1) multipliers that
*          define the matrix L from the LU factorization of A.
*
*  D       (input/output) COMPLEX*16 array, dimension (N)
*          On entry, D must contain the diagonal elements of A.
*          On exit, D is overwritten by the n diagonal elements of the
*          upper triangular matrix U from the LU factorization of A.
*
*  DU      (input/output) COMPLEX*16 array, dimension (N-1)
*          On entry, DU must contain the (n-1) superdiagonal elements
*          of A.
*          On exit, DU is overwritten by the (n-1) elements of the first
*          superdiagonal of U.
*
*  DU2     (output) COMPLEX*16 array, dimension (N-2)
*          On exit, DU2 is overwritten by the (n-2) elements of the
*          second superdiagonal of U.
*
*  IPIV    (output) INTEGER array, dimension (N)
*          The pivot indices; for 1 <= i <= n, row i of the matrix was
*          interchanged with row IPIV(i).  IPIV(i) will always be either
*          i or i+1; IPIV(i) = i indicates a row interchange was not
*          required.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
*                has been completed, but the factor U is exactly
*                singular, and division by zero will occur if it is used
*                to solve a system of equations.
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I
      COMPLEX*16         FACT, TEMP, ZDUM
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, DIMAG
*     ..
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     ..
*     .. Parameters ..
      COMPLEX*16         CZERO
      PARAMETER          ( CZERO = ( 0.0D+0, 0.0D+0 ) )
*     ..
*     .. Statement Functions ..
      DOUBLE PRECISION   CABS1
*     ..
*     .. Statement Function definitions ..
      CABS1( ZDUM ) = ABS( DBLE( ZDUM ) ) + ABS( DIMAG( ZDUM ) )
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      IF( N.LT.0 ) THEN
         INFO = -1
         CALL XERBLA( 'ZGTTRF', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
*     Initialize IPIV(i) = i
*
      DO 10 I = 1, N
         IPIV( I ) = I
   10 CONTINUE
*
      DO 20 I = 1, N - 1
         IF( DL( I ).EQ.CZERO ) THEN
*
*           Subdiagonal is zero, no elimination is required.
*
            IF( D( I ).EQ.CZERO .AND. INFO.EQ.0 )
     $         INFO = I
            IF( I.LT.N-1 )
     $         DU2( I ) = CZERO
         ELSE IF( CABS1( D( I ) ).GE.CABS1( DL( I ) ) ) THEN
*
*           No row interchange required, eliminate DL(I)
*
            FACT = DL( I ) / D( I )
            DL( I ) = FACT
            D( I+1 ) = D( I+1 ) - FACT*DU( I )
            IF( I.LT.N-1 )
     $         DU2( I ) = CZERO
         ELSE
*
*           Interchange rows I and I+1, eliminate DL(I)
*
            FACT = D( I ) / DL( I )
            D( I ) = DL( I )
            DL( I ) = FACT
            TEMP = DU( I )
            DU( I ) = D( I+1 )
            D( I+1 ) = TEMP - FACT*D( I+1 )
            IF( I.LT.N-1 ) THEN
               DU2( I ) = DU( I+1 )
               DU( I+1 ) = -FACT*DU( I+1 )
            END IF
            IPIV( I ) = IPIV( I ) + 1
         END IF
   20 CONTINUE
      IF( D( N ).EQ.CZERO .AND. INFO.EQ.0 ) THEN
         INFO = N
         RETURN
      END IF
*
      RETURN
*
*     End of ZGTTRF
*
      END
