      SUBROUTINE ZGTTRS( TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB,
     $                   INFO )
*
*  -- LAPACK routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     September 30, 1994
*
*     .. Scalar Arguments ..
      CHARACTER          TRANS
      INTEGER            INFO, LDB, N, NRHS
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      COMPLEX*16         B( LDB, * ), D( * ), DL( * ), DU( * ), DU2( * )
*     ..
*
*  Purpose
*  =======
*
*  ZGTTRS solves one of the systems of equations
*     A * X = B,  A**T * X = B,  or  A**H * X = B,
*  with a tridiagonal matrix A using the LU factorization computed
*  by ZGTTRF.
*
*  Arguments
*  =========
*
*  TRANS   (input) CHARACTER
*          Specifies the form of the system of equations:
*          = 'N':  A * X = B     (No transpose)
*          = 'T':  A**T * X = B  (Transpose)
*          = 'C':  A**H * X = B  (Conjugate transpose)
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  NRHS    (input) INTEGER
*          The number of right hand sides, i.e., the number of columns
*          of the matrix B.  NRHS >= 0.
*
*  DL      (input) COMPLEX*16 array, dimension (N-1)
*          The (n-1) multipliers that define the matrix L from the
*          LU factorization of A.
*
*  D       (input) COMPLEX*16 array, dimension (N)
*          The n diagonal elements of the upper triangular matrix U from
*          the LU factorization of A.
*
*  DU      (input) COMPLEX*16 array, dimension (N-1)
*          The (n-1) elements of the first superdiagonal of U.
*
*  DU2     (input) COMPLEX*16 array, dimension (N-2)
*          The (n-2) elements of the second superdiagonal of U.
*
*  IPIV    (input) INTEGER array, dimension (N)
*          The pivot indices; for 1 <= i <= n, row i of the matrix was
*          interchanged with row IPIV(i).  IPIV(i) will always be either
*          i or i+1; IPIV(i) = i indicates a row interchange was not
*          required.
*
*  B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
*          On entry, the right hand side matrix B.
*          On exit, B is overwritten by the solution matrix X.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================
*
*     .. Local Scalars ..
      LOGICAL            NOTRAN
      INTEGER            I, J
      COMPLEX*16         TEMP
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DCONJG, MAX
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      NOTRAN = LSAME( TRANS, 'N' )
      IF( .NOT.NOTRAN .AND. .NOT.LSAME( TRANS, 'T' ) .AND. .NOT.
     $    LSAME( TRANS, 'C' ) ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( NRHS.LT.0 ) THEN
         INFO = -3
      ELSE IF( LDB.LT.MAX( N, 1 ) ) THEN
         INFO = -10
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'ZGTTRS', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( N.EQ.0 .OR. NRHS.EQ.0 )
     $   RETURN
*
      IF( NOTRAN ) THEN
*
*        Solve A*X = B using the LU factorization of A,
*        overwriting each right hand side vector with its solution.
*
         DO 30 J = 1, NRHS
*
*           Solve L*x = b.
*
            DO 10 I = 1, N - 1
               IF( IPIV( I ).EQ.I ) THEN
                  B( I+1, J ) = B( I+1, J ) - DL( I )*B( I, J )
               ELSE
                  TEMP = B( I, J )
                  B( I, J ) = B( I+1, J )
                  B( I+1, J ) = TEMP - DL( I )*B( I, J )
               END IF
   10       CONTINUE
*
*           Solve U*x = b.
*
            B( N, J ) = B( N, J ) / D( N )
            IF( N.GT.1 )
     $         B( N-1, J ) = ( B( N-1, J )-DU( N-1 )*B( N, J ) ) /
     $                       D( N-1 )
            DO 20 I = N - 2, 1, -1
               B( I, J ) = ( B( I, J )-DU( I )*B( I+1, J )-DU2( I )*
     $                     B( I+2, J ) ) / D( I )
   20       CONTINUE
   30    CONTINUE
      ELSE IF( LSAME( TRANS, 'T' ) ) THEN
*
*        Solve A**T * X = B.
*
         DO 60 J = 1, NRHS
*
*           Solve U**T * x = b.
*
            B( 1, J ) = B( 1, J ) / D( 1 )
            IF( N.GT.1 )
     $         B( 2, J ) = ( B( 2, J )-DU( 1 )*B( 1, J ) ) / D( 2 )
            DO 40 I = 3, N
               B( I, J ) = ( B( I, J )-DU( I-1 )*B( I-1, J )-DU2( I-2 )*
     $                     B( I-2, J ) ) / D( I )
   40       CONTINUE
*
*           Solve L**T * x = b.
*
            DO 50 I = N - 1, 1, -1
               IF( IPIV( I ).EQ.I ) THEN
                  B( I, J ) = B( I, J ) - DL( I )*B( I+1, J )
               ELSE
                  TEMP = B( I+1, J )
                  B( I+1, J ) = B( I, J ) - DL( I )*TEMP
                  B( I, J ) = TEMP
               END IF
   50       CONTINUE
   60    CONTINUE
      ELSE
*
*        Solve A**H * X = B.
*
         DO 90 J = 1, NRHS
*
*           Solve U**H * x = b.
*
            B( 1, J ) = B( 1, J ) / DCONJG( D( 1 ) )
            IF( N.GT.1 )
     $         B( 2, J ) = ( B( 2, J )-DCONJG( DU( 1 ) )*B( 1, J ) ) /
     $                     DCONJG( D( 2 ) )
            DO 70 I = 3, N
               B( I, J ) = ( B( I, J )-DCONJG( DU( I-1 ) )*B( I-1, J )-
     $                     DCONJG( DU2( I-2 ) )*B( I-2, J ) ) /
     $                     DCONJG( D( I ) )
   70       CONTINUE
*
*           Solve L**H * x = b.
*
            DO 80 I = N - 1, 1, -1
               IF( IPIV( I ).EQ.I ) THEN
                  B( I, J ) = B( I, J ) - DCONJG( DL( I ) )*B( I+1, J )
               ELSE
                  TEMP = B( I+1, J )
                  B( I+1, J ) = B( I, J ) - DCONJG( DL( I ) )*TEMP
                  B( I, J ) = TEMP
               END IF
   80       CONTINUE
   90    CONTINUE
      END IF
*
*     End of ZGTTRS
*
      END
